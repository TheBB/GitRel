from __future__ import annotations

import os
import re
import shutil
import sys
import stat
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING, Any, Self, Literal

import click
import httpx
import inquirer
from github import Auth, Github
from pydantic import BaseModel, Field, field_serializer, field_validator
from semantic_version import Version
from tqdm import tqdm
from xdg_base_dirs import xdg_config_home, xdg_data_home

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType


def parse_version(input: str) -> Version:
    return Version.coerce(input)


def is_executable(path: Path) -> bool:
    output = run(["file", str(path)], capture_output=True)
    output.check_returncode()
    return b"executable" in output.stdout


def ignore_file(name: str) -> bool:
    name = name.lower()
    return name.endswith(".md") or name in ["license"]


@dataclass(slots=True)
class VersionConstraint(ABC):
    bound: Version

    @abstractmethod
    def matches(self, version: Version) -> bool: ...


class VersionHigher(VersionConstraint):
    def matches(self, version: Version) -> bool:
        return version > self.bound


class VersionLower(VersionConstraint):
    def matches(self, version: Version) -> bool:
        return version < self.bound


class VersionHigherOrEqual(VersionConstraint):
    def matches(self, version: Version) -> bool:
        return version >= self.bound


class VersionLowerOrEqual(VersionConstraint):
    def matches(self, version: Version) -> bool:
        return version <= self.bound


class VersionEqual(VersionConstraint):
    def matches(self, version: Version) -> bool:
        return version == self.bound


class Manager:
    bin_path: Path = Path.home() / ".local" / "bin"
    data_path: Path = xdg_data_home() / "gitrel"
    store_path: Path = data_path / "store"
    config_path: Path = xdg_config_home() / "gitrel"

    config: Config
    github: Github
    state: State

    def __init__(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.config_path.mkdir(parents=True, exist_ok=True)

        with (self.config_path / "config.json").open("r") as f:
            self.config = Config.model_validate_json(f.read())
        self.github = self.config.github()

    def __enter__(self) -> Self:
        with (self.data_path / "state.json").open("r") as f:
            self.state = StateLoader.construct(f.read()).finalize()
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with (self.data_path / "state.json").open("w") as f:
            f.write(self.state.model_dump_json())

    def installed_packages(self) -> Iterator[Package]:
        yield from self.state.packages.values()

    def get_package(self, binary_or_package: str) -> Package:
        if binary_or_package in self.state.binary_to_package:
            return self.state.packages[self.state.binary_to_package[binary_or_package]]
        return self.state.packages[binary_or_package]

    def versions(self, spec: PackageSpec) -> list[tuple[int | str, Version]]:
        repo = self.github.get_repo(str(spec.repo))

        try:
            releases = list(repo.get_releases())
        except StopIteration:
            print(f"Unable to find any releases for {repo}", file=sys.stderr)
            sys.exit(1)

        versions: list[tuple[int | str, Version]] = []
        for release in releases:
            try:
                version = Version.coerce(release.title.removeprefix("v"))
            except ValueError:
                continue
            if all(constraint.matches(version) for constraint in spec.constraints):
                versions.append((release.id, version))

        return versions

    def install(
        self,
        repo_name: GithubRepo,
        version: Version,
        release_id: int | str,
        preferred_asset: str | None = None,
        uninstall_existing: bool = False,
    ) -> None:
        repo = self.github.get_repo(str(repo_name))
        release = repo.get_release(release_id)

        # Pick which release asset to use
        assets = {asset.name: asset for asset in release.assets}
        if preferred_asset and preferred_asset in assets:
            asset = assets[preferred_asset]
        else:
            if preferred_asset:
                print(f"Unable to find asset {preferred_asset}")
            answer = inquirer.prompt(
                [
                    inquirer.List(
                        "asset_name",
                        message="Which asset to use?",
                        choices=assets.keys(),
                    )
                ]
            )

            assert answer is not None, "Aborting"
            asset = assets[answer["asset_name"]]

        # Download asset to temp path
        temp_path = self.store_path / "temp"
        temp_store_path = self.store_path / f"NEW-{repo_name.store_path()}"
        final_store_path = self.store_path / repo_name.store_path()

        with (
            httpx.stream("GET", asset.browser_download_url, follow_redirects=True) as r,
            tqdm(desc="Downloading", total=asset.size, unit="B", unit_scale=True) as p,
            temp_path.open("wb") as f,
        ):
            for chunk in r.iter_bytes():
                p.update(len(chunk))
                f.write(chunk)

        # Interpret asset file
        try:
            if asset.name.endswith(".tar.gz"):
                with tarfile.open(temp_path) as t:
                    candidates = [member.name for member in t.getmembers() if not ignore_file(member.name)]
                    t.extractall(path=temp_store_path)
            elif is_executable(temp_path):
                temp_store_path.mkdir()
                temp_path.rename(temp_store_path / asset.name)
                candidates = [asset.name]
            else:
                assert False, "Unable to understand asset file"
        finally:
            temp_path.unlink(missing_ok=True)

        # Find binaries
        try:
            candidates = [c for c in candidates if is_executable(temp_store_path / c)]
            assert candidates, "Unable to find any binaries"
        except:
            shutil.rmtree(temp_store_path)
            raise

        # Uninstall existing package if required
        if uninstall_existing:
            try:
                self.uninstall(self.state.packages[str(repo_name)])
            except:
                shutil.rmtree(temp_store_path)
                raise

        # Replace store path
        shutil.move(temp_store_path, final_store_path)

        package = Package(
            repo=repo_name,
            current_version=version,
            installed_from_filename=asset.name,
            store_path=repo_name.store_path(),
        )

        for c in candidates:
            print(f"Binary: {c}")
            bin_name = input("Install as: ").strip()
            if not bin_name:
                continue
            os.symlink(final_store_path / c, self.bin_path / bin_name)
            (final_store_path / c).chmod(stat.S_IRWXU)
            package.binaries[bin_name] = c
            self.state.binary_to_package[bin_name] = str(repo_name)

        self.state.packages[str(repo_name)] = package

    def uninstall(self, package: Package) -> None:
        for binary in package.binaries:
            (self.bin_path / binary).unlink()
            del self.state.binary_to_package[binary]

        shutil.rmtree(self.store_path / package.store_path)
        del self.state.packages[str(package.repo)]


class Package_V1(BaseModel, arbitrary_types_allowed=True):
    binaries: list[str] = Field(default=[])
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

    def upgrade(self) -> Package_V2:
        data = self.model_dump()
        data["binaries"] = {name: name for name in self.binaries}
        return Package_V2.model_validate(data)


class Package_V2(BaseModel, arbitrary_types_allowed=True):
    binaries: dict[str, str] = Field(default={})
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


class State_V1(BaseModel):
    version: Literal[1]
    packages: dict[str, Package_V1] = Field(default_factory=dict)
    binary_to_package: dict[str, str] = Field(default_factory=dict)

    def upgrade(self) -> State_V2:
        data = self.model_dump()
        data["packages"] = {name: package.upgrade() for name, package in self.packages.items()}
        data["version"] = 2
        return State_V2.model_validate(data)


class State_V2(BaseModel):
    version: Literal[2]
    packages: dict[str, Package_V2] = Field(default_factory=dict)
    binary_to_package: dict[str, str] = Field(default_factory=dict)


Package = Package_V2
State = State_V2


class StateLoader(BaseModel):
    data: State_V1 | State_V2

    @classmethod
    def construct(cls, json_data: str) -> StateLoader:
        return cls.model_validate_json(f'{{"data": {json_data}}}')

    def finalize(self) -> State:
        if isinstance(self.data, State_V2):
            return self.data
        if isinstance(self.data, State_V1):
            return self.data.upgrade()
        assert False


class Config(BaseModel):
    token: str

    def github(self) -> Github:
        auth = Auth.Token(self.token)
        return Github(auth=auth)


class GithubRepo(BaseModel):
    org: str
    repo: str

    def __str__(self) -> str:
        return f"{self.org}/{self.repo}"

    def store_path(self) -> str:
        return f"{self.org}-{self.repo}"


@dataclass
class PackageSpec:
    repo: GithubRepo
    constraints: list[VersionConstraint] = field(default_factory=list)


class PackageSpecType(click.ParamType):
    name = "org/repo"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> PackageSpec:
        try:
            assert isinstance(value, str)
            version_pattern = Version.partial_version_re.pattern[1:-1]
            regex = rf"^(?P<org>[\w-]+)/(?P<repo>[\w-]+)(?P<constraints>((@|>|<|>=|<=){version_pattern})*)$"
            match = re.match(regex, value)
            assert match is not None

            spec = PackageSpec(repo=GithubRepo(org=match.group("org"), repo=match.group("repo")))

            constraints = match["constraints"]
            while constraints:
                match = re.match(
                    rf"^(?P<operator>(@|\>|\<|>=|<=))(?P<version>{version_pattern})", constraints
                )
                assert match is not None
                constraints = constraints[match.end() :]

                version = Version.coerce(match.group("version"))
                match match.group("operator"):
                    case ">":
                        spec.constraints.append(VersionHigher(version))
                    case "<":
                        spec.constraints.append(VersionLower(version))
                    case ">=":
                        spec.constraints.append(VersionHigherOrEqual(version))
                    case "<=":
                        spec.constraints.append(VersionLowerOrEqual(version))
                    case "@":
                        spec.constraints.append(VersionEqual(version))

            return spec
        except (ValueError, AssertionError):
            self.fail(f"{value!r} is not a valid github repo string", param, ctx)


@click.group
def main() -> None:
    path = Manager.config_path / "config.json"
    if not path.exists():
        token = input("Access token: ").strip()
        config = Config(token=token)
        with path.open("w") as f:
            f.write(config.model_dump_json())


@main.command("list")
def list_() -> None:
    with Manager() as manager:
        for package in manager.installed_packages():
            print(f"{package.repo} @ {package.current_version}")
            for binary in package.binaries:
                print(f"    {binary}")


@main.command
@click.argument("spec", type=PackageSpecType())
def install(spec: PackageSpec) -> None:
    with Manager() as manager:
        versions = manager.versions(spec)
        if not versions:
            print("Unable to find any matching releases", file=sys.stderr)
            sys.exit(1)

        release_id, version = versions[0]
        print(f"Installing version {version}")

        try:
            manager.install(spec.repo, version, release_id)
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(2)


@main.command
@click.argument("package_name", type=str)
def remove(package_name: str) -> None:
    with Manager() as manager:
        try:
            package = manager.get_package(package_name)
        except KeyError:
            print(f"Package not installed: {package_name}", file=sys.stderr)
            sys.exit(1)

        manager.uninstall(package)
        binaries = ", ".join(package.binaries)
        print(f"Removed binaries: {binaries}")


@main.command
@click.argument("package_name", type=str)
def upgrade(package_name: str) -> None:
    with Manager() as manager:
        try:
            package = manager.get_package(package_name)
        except KeyError:
            print(f"Package not installed: {package_name}")

        versions = manager.versions(PackageSpec(package.repo, [VersionHigher(package.current_version)]))
        if not versions:
            print(f"Unable to find a newer version than current ({package.current_version})", file=sys.stderr)
            sys.exit(0)

        release_id, version = versions[0]
        print(f"Installing version {version} over {package.current_version}")

        try:
            manager.install(
                package.repo,
                version,
                release_id,
                preferred_asset=package.installed_from_filename,
                uninstall_existing=True,
            )
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(2)
