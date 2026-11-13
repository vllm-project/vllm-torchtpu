# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Single-backbone runtime 20L / 40L conditional execution (Exp 7)."""

import collections
from collections.abc import Sequence
import re
from typing import Any

import jax
import torch
from torch.tpu import pallas
from vllm.logger import init_logger

import vllm_torchtpu.envs as envs

logger = init_logger(__name__)

_HOOK_INSTALLED = False


def _dsv41_cond_upper_boundary_fn(
    hs: jax.Array, pre_mix: jax.Array, flag: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return hs, pre_mix


_dsv41_cond_upper_boundary_op = pallas.jax_op(
    "tpu::dsv41_cond_upper_boundary", _dsv41_cond_upper_boundary_fn
)


@torch.library.register_fake("tpu::dsv41_cond_upper_boundary")
def _dsv41_cond_upper_boundary_fake(
    hs: torch.Tensor, pre_mix: torch.Tensor, flag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hs), torch.empty_like(pre_mix)


def _inject_donor_attrs_in_header(header: str, donated_inputs: set[int]) -> str:
    if not donated_inputs:
        return header

    def _repl(m: re.Match[str]) -> str:
        aname, idx_str, atype, attr = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4) or "",
        )
        if int(idx_str) not in donated_inputs or "buffer_donor" in attr:
            return m.group(0)
        if attr.strip():
            merged = attr.strip()[:-1] + ", jax.buffer_donor = true}"
            return f"{aname}: {atype} {merged}"
        return f"{aname}: {atype} {{jax.buffer_donor = true}}"

    return re.sub(r"(%arg(\d+)):\s*(tensor<[^>]+>)(\s*\{[^}]*\})?", _repl, header)


def _rewrite_dsv41_cond_mlir(
    mlir_text: str, donated_inputs: Sequence[int] | None = None
) -> str:
    if "dsv41_cond_upper_boundary" not in mlir_text:
        return mlir_text

    donor_indices = set(donated_inputs or ())
    lines = mlir_text.splitlines()
    main_start = None
    main_end = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("func.func @main("):
            main_start = idx
        elif main_start is not None and line == "  }":
            main_end = idx
            break
    assert main_start is not None and main_end is not None

    header = _inject_donor_attrs_in_header(lines[main_start], donor_indices)
    arg_types: dict[str, str] = {}
    donor_args: set[str] = set()
    for m in re.finditer(r"(%arg(\d+)):\s*(tensor<[^>]+>)(\s*\{[^}]*\})?", header):
        aname, idx_str, atype, attr = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4) or "",
        )
        arg_types[aname] = atype
        if "buffer_donor" in attr or int(idx_str) in donor_indices:
            donor_args.add(aname)

    ret_line = lines[main_end - 1].strip()
    assert ret_line.startswith("return ")
    ret_body = ret_line[len("return ") :]
    ret_vals_str, ret_types_str = ret_body.split(" : ", 1)
    ret_vals = [v.strip() for v in ret_vals_str.split(",")]
    ret_types = [t.strip() for t in ret_types_str.split(",")]

    body_lines = lines[main_start + 1 : main_end - 1]
    op_chunks: list[list[str]] = []
    for bline in body_lines:
        if bline.startswith("    %"):
            op_chunks.append([bline])
        else:
            assert op_chunks, f"Unexpected leading MLIR line: {bline}"
            op_chunks[-1].append(bline)

    ops: list[tuple[list[str], list[str], list[str]]] = []
    gate_val: str | None = None
    pred_arg: str | None = None
    ssa_re = re.compile(r"%[A-Za-z0-9_]+")

    for chunk in op_chunks:
        first_line = chunk[0]
        lhs_rhs = first_line.split(" = ", 1)
        assert len(lhs_rhs) == 2, f"Unexpected MLIR op line: {first_line}"
        lhs, rhs_first = lhs_rhs[0].strip(), lhs_rhs[1]
        full_rhs = rhs_first + ("\n" + "\n".join(chunk[1:]) if len(chunk) > 1 else "")
        defs = ssa_re.findall(lhs)
        uses = ssa_re.findall(full_rhs)
        if "dsv41_cond_upper_boundary" in first_line:
            gate_val = defs[0]
            pred_arg = uses[2]
        ops.append((chunk, defs, uses))

    assert gate_val is not None and pred_arg is not None

    # Exact forward transitive closure from gate_val -> Decoder set D
    in_dec: set[str] = {gate_val}
    dec_op_indices: set[int] = set()
    for idx, (_chunk, defs, uses) in enumerate(ops):
        if any(u in in_dec for u in uses):
            dec_op_indices.add(idx)
            in_dec.update(defs)

    def _base_ssa(v: str) -> str:
        return v.split("#", 1)[0]

    dec_ret_indices = [i for i, rv in enumerate(ret_vals) if _base_ssa(rv) in in_dec]
    def_to_uses: dict[str, list[str]] = {}
    for _chunk, defs, uses in ops:
        for d in defs:
            def_to_uses[d] = uses

    def _shortest_donor_candidates(rv: str, rtype: str) -> list[tuple[int, str]]:
        start = _base_ssa(rv)
        q: collections.deque[tuple[str, int]] = collections.deque([(start, 0)])
        seen: set[str] = {start}
        found: list[tuple[int, str]] = []
        while q:
            cur, dist = q.popleft()
            if cur in donor_args and arg_types.get(cur) == rtype:
                found.append((dist, cur))
                continue
            if cur in def_to_uses and cur in in_dec:
                for nxt in def_to_uses[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, dist + 1))
        return found

    all_pairs: list[tuple[int, int, str]] = []
    for k, i in enumerate(dec_ret_indices):
        rv, rtype = ret_vals[i], ret_types[i]
        for dist, dname in _shortest_donor_candidates(rv, rtype):
            all_pairs.append((dist, k, dname))
    all_pairs.sort()

    assigned_donor_for_k: dict[int, str] = {}
    used_donors: set[str] = set()
    for _dist, k, dname in all_pairs:
        if k not in assigned_donor_for_k and dname not in used_donors:
            assigned_donor_for_k[k] = dname
            used_donors.add(dname)

    enc_lines = [
        line for i in range(len(ops)) if i not in dec_op_indices for line in ops[i][0]
    ]
    dec_lines = [
        line for i in range(len(ops)) if i in dec_op_indices for line in ops[i][0]
    ]

    if_ret_types = [ret_types[i] for i in dec_ret_indices]
    true_ret_vals = [ret_vals[i] for i in dec_ret_indices]
    false_stmts: list[str] = []
    false_ret_vals: list[str] = []
    for k, i in enumerate(dec_ret_indices):
        rtype = ret_types[i]
        donor = assigned_donor_for_k.get(k)
        if donor is not None:
            false_ret_vals.append(donor)
        else:
            zname = f"%zero_cond_{k}"
            elem = rtype.split("x")[-1][:-1]
            zlit = "0" if elem.startswith(("i", "u")) else "0.0"
            false_stmts.append(
                f"      {zname} = stablehlo.constant dense<{zlit}> : {rtype}"
            )
            false_ret_vals.append(zname)

    if_lhs = ", ".join(f"%cond_out_{k}" for k in range(len(dec_ret_indices)))
    if_type_sig = (
        f"({', '.join(if_ret_types)})" if len(if_ret_types) != 1 else if_ret_types[0]
    )

    new_body = list(enc_lines)
    new_body.append("    %c0_cond = stablehlo.constant dense<0> : tensor<1xi32>")
    new_body.append(
        f"    %cmp_cond = stablehlo.compare NE, {pred_arg}, %c0_cond : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>"
    )
    new_body.append(
        "    %pred_cond = stablehlo.reshape %cmp_cond : (tensor<1xi1>) -> tensor<i1>"
    )
    new_body.append(f'    {if_lhs} = "stablehlo.if"(%pred_cond) ({{')
    for dl in dec_lines:
        new_body.append("  " + dl)
    new_body.append(
        f"      stablehlo.return {', '.join(true_ret_vals)} : {', '.join(if_ret_types)}"
    )
    new_body.append("    }, {")
    new_body.extend(false_stmts)
    new_body.append(
        f"      stablehlo.return {', '.join(false_ret_vals)} : {', '.join(if_ret_types)}"
    )
    new_body.append(f"    }}) : (tensor<i1>) -> {if_type_sig}")

    cond_map = {
        dec_ret_indices[k]: f"%cond_out_{k}" for k in range(len(dec_ret_indices))
    }
    final_rets = [cond_map.get(i, ret_vals[i]) for i in range(len(ret_vals))]
    new_body.append(f"    return {', '.join(final_rets)} : {ret_types_str}")

    logger.info(
        "[DSV41 SingleBackboneCond] Wrapped %d/%d ops and %d/%d outputs in stablehlo.if on %s",
        len(dec_lines),
        len(ops),
        len(dec_ret_indices),
        len(ret_vals),
        pred_arg,
    )
    return "\n".join(lines[:main_start] + [header] + new_body + lines[main_end:])


def install_dsv41_cond_mlir_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED or not envs.VLLM_DSV41_SINGLE_BACKBONE_COND:
        return

    import torch_tpu._internal.export as exp_pkg
    from torch_tpu._internal.export import export as exp_mod

    orig_fx_to_mlir = exp_mod.fx_to_mlir
    ttc = exp_mod.tpu_torch_compile

    def _cond_fx_to_mlir(
        module: torch.fx.GraphModule,
        args: Sequence[Any],
        build_mlir_module: bool = True,
        use_stablehlo_bounds: bool = False,
        argument_layouts: Sequence[Sequence[int]] | None = None,
        dynamic_outputs: Sequence[bool] | None = None,
        donated_inputs: Sequence[int] | None = None,
        module_name: str | None = None,
    ) -> Any:
        has_cond_boundary = envs.VLLM_DSV41_SINGLE_BACKBONE_COND and any(
            "dsv41_cond_upper_boundary" in str(n.target) for n in module.graph.nodes
        )
        if not has_cond_boundary:
            return orig_fx_to_mlir(
                module,
                args,
                build_mlir_module=build_mlir_module,
                use_stablehlo_bounds=use_stablehlo_bounds,
                argument_layouts=argument_layouts,
                dynamic_outputs=dynamic_outputs,
                donated_inputs=donated_inputs,
                module_name=module_name,
            )

        import fcntl
        import gc
        import hashlib
        import os

        res_orig = orig_fx_to_mlir(
            module,
            args,
            build_mlir_module=True,
            use_stablehlo_bounds=use_stablehlo_bounds,
            argument_layouts=argument_layouts,
            dynamic_outputs=dynamic_outputs,
            donated_inputs=donated_inputs,
            module_name=module_name,
        )
        raw_text = ttc.serialize_mlir_text(res_orig.module)
        orig_layouts = [list(p[0]) for p in res_orig.executable.get_parameter_layouts()]
        mlir_result_tensors = res_orig.mlir_result_tensors
        reconstruct_fx_outputs_fn = res_orig.reconstruct_fx_outputs_fn
        updates_rng = res_orig.updates_default_generator_state
        deduped_dynamic_outputs = res_orig.dynamic_outputs
        unique_output_indices = res_orig.unique_output_indices
        del res_orig
        gc.collect()

        rewritten_text = _rewrite_dsv41_cond_mlir(
            raw_text, donated_inputs=donated_inputs
        )
        del raw_text

        h = hashlib.sha256(rewritten_text.encode("utf-8")).hexdigest()[:24]
        cache_dir = envs.VLLM_XLA_CACHE_PATH
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"cond_exec_{h}.bin")
        cond_mod = ttc.parse_mlir_text(rewritten_text) if build_mlir_module else None
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as cf:
                executable = ttc.load_serialized_executable(cf.read())
            logger.info(
                "[DSV41 SingleBackboneCond] Loaded cached executable from %s",
                cache_path,
            )
        else:
            with open("/tmp/dsv41_cond_compile.lock", "w") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as cf:
                        executable = ttc.load_serialized_executable(cf.read())
                    logger.info(
                        "[DSV41 SingleBackboneCond] Loaded cached executable from %s",
                        cache_path,
                    )
                else:
                    mod_to_compile = cond_mod or ttc.parse_mlir_text(rewritten_text)
                    executable = ttc.compile_mlir(
                        mod_to_compile,
                        argument_layouts=orig_layouts,
                    )
                    del mod_to_compile
                    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
                    with open(tmp_path, "wb") as cf:
                        cf.write(ttc.serialize_executable(executable))
                    os.replace(tmp_path, cache_path)
                    logger.info(
                        "[DSV41 SingleBackboneCond] Saved compiled executable to %s",
                        cache_path,
                    )
        del rewritten_text
        gc.collect()
        return exp_mod.ExportedMlir(
            module=cond_mod if build_mlir_module else None,
            executable=executable,
            mlir_result_tensors=mlir_result_tensors,
            reconstruct_fx_outputs_fn=reconstruct_fx_outputs_fn,
            updates_default_generator_state=updates_rng,
            dynamic_outputs=deduped_dynamic_outputs,
            unique_output_indices=unique_output_indices,
        )

    exp_mod.fx_to_mlir = _cond_fx_to_mlir
    exp_pkg.fx_to_mlir = _cond_fx_to_mlir
    _HOOK_INSTALLED = True
    logger.info("[DSV41 SingleBackboneCond] Installed fx_to_mlir stablehlo.if hook")
