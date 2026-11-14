# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The `scenario` command line."""

import sys
from typing import NoReturn

import click

from scenario import PLAN_FORMAT, __version__, gen

EXIT_PROBLEMS = 1
EXIT_USAGE = 2


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__,
    "--version",
    prog_name="scenario",
    message=f"%(prog)s %(version)s (plan format {PLAN_FORMAT})",
)
def main() -> None:
    """Validate, show, render and run vLLM server scenarios."""


def _fail(message: str, code: int = EXIT_USAGE) -> NoReturn:
    click.echo(f"error: {message}", err=True)
    sys.exit(code)


# dev --------------------------------------------------------------------


@main.group()
def dev() -> None:
    """Commands for working on the tool itself."""


@dev.command()
@click.option(
    "--check", is_flag=True, help="Write nothing; fail when a generated file is stale."
)
def regen(check: bool) -> None:
    """Regenerate the JSON schemas and REFERENCE.md from the models."""
    try:
        root = gen.tool_dir()
    except FileNotFoundError as error:
        _fail(str(error))
    if check:
        names = gen.stale(root)
        for name in names:
            click.echo(f"stale: tools/scenario/{name}; run: {gen.REGEN}", err=True)
        sys.exit(EXIT_PROBLEMS if names else 0)
    for name in gen.write(root):
        click.echo(f"wrote tools/scenario/{name}")
