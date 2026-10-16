"""Held-out, independently specified Knowing-to-Acting benchmark set.

Contents
--------
``ORACLE_SPEC.md``       the written rules every label comes from (the only source of
                         truth for ``expected_action``).
``cases_heldout.py``     ``HELDOUT_CASES``: the frozen, independently authored states.
``check_heldout.py``     plain-script structural check + manifest hash.

The labels were produced by applying ``ORACLE_SPEC.md`` to frozen states by hand. No
decision engine (``bench/battle.py`` runners, the ``Governor``, the ``OpenDecider``, the
candidate builder) was executed to produce them, and no runtime module is imported by
the data or the checker.
"""

from __future__ import annotations