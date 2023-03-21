
import os, pathlib

from dynaconf import Dynaconf

settings = Dynaconf(
    environments=True,
    envvar_prefix="DYNACONF",
    env="pp-mattei",
    settings_files=['settings.toml', '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
if not os.path.isfile(os.path.join(os.getcwd(), "settings.toml")):
    print(__file__)
    print(os.getcwd())
    raise FileNotFoundError (
        "settings.toml is missing in the current directory \n" + 
        "install a template by executing 'ocean_forest.install_template()'")
