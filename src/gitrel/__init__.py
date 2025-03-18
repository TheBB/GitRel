from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path
from typing import Annotated, Any, Self
from subprocess import run

import click
import httpx
import inquirer
from github import Auth, Github
from pydantic import BaseModel, BeforeValidator, Field, field_validator, field_serializer
from semantic_version import Version
from tqdm import tqdm
from xdg_base_dirs import xdg_config_home, xdg_data_home

BIN = Path.home() / ".local" / "bin"
DATA = xdg_data_home() / "gitrel"
STORE = DATA / "store"
CONFIG = xdg_config_home() / "gitrel"

STORE.mkdir(parents=True, exist_ok=True)
CONFIG.mkdir(parents=True, exist_ok=True)


def parse_version(input: str) -> Version:
    return Version.coerce(input)


def is_executable(path: Path) -> bool:
    output = run(["file", str(path)], capture_output=True)
    output.check_returncode()
    return b"executable" in output.stdout


PdVersion = Annotated[Version, BeforeValidator(parse_version)]


class GithubRepo(BaseModel):
    org: str
    repo: str

    def __str__(self) -> str:
        return f"{self.org}/{self.repo}"

    def store_path(self) -> str:
        return f"{self.org}-{self.repo}"


class Package(BaseModel, arbitrary_types_allowed=True):
    binaries: list[str]
    repo: GithubRepo
    current_version: Version
    installed_from_filename: str
    store_path: str

    @field_validator("current_version", mode="before")
    def parse_version(cls, value: Any) -> Version:
        if isinstance(value, Version):
            return value
        if isinstance(value, str):
            return Version.coerce(value)
        raise ValueError(f"Invalid version: {value}")

    @field_serializer("current_version")
    def serialize_version(self, value: Version) -> str:
        return str(value)


class State(BaseModel):
    packages: dict[str, Package] = Field(default_factory=dict)
    binary_to_package: dict[str, str] = Field(default_factory=dict)

    @staticmethod
    def load() -> State:
        path = DATA / "state.json"
        if not path.exists():
            return State()
        with path.open("r") as f:
            data = f.read()
        return State.model_validate_json(data)

    def save(self):
        path = DATA / "state.json"
        with path.open("w") as f:
            f.write(self.model_dump_json())


class Config(BaseModel):
    token: str

    def github(self) -> Github:
        auth = Auth.Token(self.token)
        return Github(auth=auth)


class OrgRepo(click.ParamType):
    name = "org/repo"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> GithubRepo:
        try:
            assert isinstance(value, str)
            org, repo = value.split("/")
            assert org and repo
            return GithubRepo(org=org, repo=repo)
        except (ValueError, AssertionError):
            self.fail(f"{value!r} is not a valid github repo string", param, ctx)


@click.group
@click.pass_context
def main(ctx: click.Context) -> None:
    path = CONFIG / "config.json"
    if not path.exists():
        token = input("Access token: ").strip()
        config = Config(token=token)
        with path.open("w") as f:
            f.write(config.model_dump_json())
    else:
        with path.open("r") as f:
            data = f.read()
        config = Config.model_validate_json(data)
    ctx.obj = config


@main.command
def list() -> None:
    print("Hello there")


@main.command
@click.argument("repo_name", type=OrgRepo())
@click.pass_context
def install(ctx: click.Context, repo_name: GithubRepo) -> None:
    config: Config = ctx.obj
    github = config.github()

    repo = github.get_repo(str(repo_name))

    try:
        release = next(iter(repo.get_releases()))
    except StopIteration:
        print(f"Unable to find any releases for {repo}", file=sys.stderr)
        sys.exit(1)

    version_string = release.title.removeprefix("v")
    try:
        version = Version.coerce(version_string)
    except ValueError:
        print(f"Found a release, but was unable to understand the version '{version_string}", file=sys.stderr)
        sys.exit(2)

    print(f"Using version {version}")
    assets = {asset.name: {"url": asset.browser_download_url, "size": asset.size} for asset in release.assets}
    answer = inquirer.prompt(
        [
            inquirer.List(
                "asset",
                message="Which asset to use?",
                choices=assets.keys(),
            )
        ]
    )
    if answer is None:
        print("Aborting", file=sys.stderr)
        sys.exit(3)

    asset_name: str = answer["asset"]
    url = assets[asset_name]["url"]
    size = assets[asset_name]["size"]
    temp_path = STORE / "temp"
    store_path = STORE / repo_name.store_path()

    with (
        httpx.stream("GET", url, follow_redirects=True) as r,
        temp_path.open("wb") as f,
        tqdm(desc="Downloading", total=size, unit="B", unit_scale=True) as p,
    ):
        for chunk in r.iter_bytes():
            p.update(len(chunk))
            f.write(chunk)

    if asset_name.endswith(".tar.gz"):
        with tarfile.open(temp_path) as t:
            candidates = [
                member.name
                for member in t.getmembers()
                if not member.name.endswith(".md") and member.name not in ["LICENSE"]
            ]
            t.extractall(path=STORE / repo_name.store_path())

    temp_path.unlink()
    candidates = [c for c in candidates if is_executable(store_path / c)]

    if len(candidates) < 1:
        print("Unable to find an executable file", file=sys.stderr)
        sys.exit(4)

    if len(candidates) > 1:
        bin_name = inquirer.prompt(
            [
                inquirer.List(
                    "member",
                    message="Which binary to use?",
                    choices=candidates,
                )
            ]
        )
        if bin_name is None:
            print("Aborting")
            sys.exit(5)
        bin_name = bin_name["member"]
    else:
        bin_name = candidates[0]

    os.symlink(store_path / bin_name, BIN / bin_name)
    print(f"Installed binary: {bin_name}")

    state = State.load()
    assert bin_name not in state.packages
    state.packages[str(repo_name)] = Package(
        binaries=[bin_name],
        repo=repo_name,
        current_version=version,
        installed_from_filename=asset_name,
        store_path=store_path.name,
    )

    state.binary_to_package[bin_name] = str(repo_name)

    state.save()


@main.command
def remove() -> None:
    pass


@main.command
def upgrade() -> None:
    pass
