# Downloading Sentinel data
------------

1. Set up a conda environment using the `envs/sentinel.yml` file as follows:
`conda env create -f sentinel.yml`
2. Activate the environment with `conda activate sentinel`
3. Register for a google earth engine account [here](https://signup.earthengine.google.com/#!/)
4. Install `gcloud` following the instructions [here](https://cloud.google.com/sdk/docs/install)
5. Authenticate earth engine with the command `earthengine authenticate`. NOTE: If using a cluster / remote machine, this should be done with the `--quiet` flag, i.e. `earthengine authenticate --quiet`.
6. Check it's working by running a Python interpreter with the commands `import ee` and then `ee.Initialize()`. If no errors are thrown then set-up is complete.
