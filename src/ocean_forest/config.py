
import os, pathlib, shutil

from dynaconf import Dynaconf

settings = Dynaconf(
    environments=True,
    envvar_prefix="DYNACONF",
    env="pp-mattei",
    settings_files=['settings.toml', '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
srcfile = pathlib.Path(__file__).parent / "settings.toml"
destfile = os.path.join(os.getcwd(), "settings.toml")
if not os.path.isfile(destfile):
    shutil.copyfile(srcfile, destfile)
    raise FileNotFoundError (
        "settings.toml is missing in the current directory \n" + 
        "A template is copied here, please edit as necessary.")
