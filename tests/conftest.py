"""Shared test configuration: hypothesis profiles.

Select with HYPOTHESIS_PROFILE=ci (or --hypothesis-profile); the default
``dev`` profile keeps local runs fast while CI explores more examples.
"""

import os

from hypothesis import settings

settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
