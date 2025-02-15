If you want to add a new set of environments.

1. Create a environment wrapper that provides following function:

2. Create an builder that builds from a configuration string by calling environment wrapper:

3. Register the environment in `env_builder.py`.

4. run `test_env.py` to make sure it works.