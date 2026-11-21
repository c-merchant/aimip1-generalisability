"""
Unit testing for AMIP loader functionality
"""

from code import data
from code.data.model_loader import AMIP_SCENARIOS
import pytest


def test_amip_scenario_map():
    assert AMIP_SCENARIOS['amip'] == '0K', "amip should map to 0K."
    assert AMIP_SCENARIOS['amip-p4K'] == '4K', "amip-p4K should map to 4K."
    assert AMIP_SCENARIOS['amip-m4K'] == '-4K', "amip-m4K should map to -4K."
    # Unmapped scenario dirs (amip-future4K) should be excluded, not passed through raw.
    assert 'amip-future4K' not in AMIP_SCENARIOS, "amip-future4K should not be in the scenario map."


def test_amip_loader_invalid_path():
    with pytest.raises(FileNotFoundError, match="Path not found"):
        data.load_amip('/nonexistent/path')
