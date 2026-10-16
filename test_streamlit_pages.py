"""Streamlit Multi-Page & Cloud Standalone Verification Test."""

import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from streamlit.testing.v1 import AppTest

DATA_PATH = PROJECT_ROOT / "data" / "energydata_complete.csv"
BACKUP_PATH = Path("/tmp/energydata_complete_backup.csv")


def test_page(page_path: str, title: str):
    print(f"\n--- Testing Page: {title} ({page_path}) ---")
    at = AppTest.from_file(page_path)
    at.run(timeout=20)
    if at.exception:
        for ex in at.exception:
            print("ERROR in page:", ex.message)
        raise RuntimeError(f"Exceptions found in {page_path}")
    print(f"✅ {title} loaded successfully without exceptions!")
    return at


def test_results_page_interactions():
    print("\n--- Testing Results Page Interactive Toggles ---")
    at = AppTest.from_file("app/pages/results.py")
    at.run(timeout=20)
    assert not at.exception, f"Initial run exception: {at.exception}"
    print("✅ Results page initial render (Default RMSE) OK")

    # Toggle radio to MAE
    if at.radio:
        print("Toggling metric to MAE...")
        at.radio[0].set_value("MAE").run()
        assert not at.exception, f"MAE toggle exception: {at.exception}"
        print("✅ Results page MAE toggle OK")

        # Toggle radio to R2
        print("Toggling metric to R2...")
        at.radio[0].set_value("R2").run()
        assert not at.exception, f"R2 toggle exception: {at.exception}"
        print("✅ Results page R2 toggle OK")
    else:
        print("Note: at.radio not found in first level elements.")


def test_cloud_standalone_mode():
    print("\n" + "=" * 60)
    print("Simulating Streamlit Cloud Standalone Mode (No raw Kaggle CSV)")
    print("=" * 60)
    assert DATA_PATH.exists(), f"Raw data file missing before test: {DATA_PATH}"
    original_size = DATA_PATH.stat().st_size

    try:
        # Move raw CSV to temporary backup
        print(f"Moving {DATA_PATH} -> {BACKUP_PATH} ...")
        shutil.move(str(DATA_PATH), str(BACKUP_PATH))
        assert not DATA_PATH.exists(), "DATA_PATH should not exist during standalone test!"
        print("Dataset moved. DATA_PATH.exists() is False.")

        # Test Dataset page
        print("Running app/pages/dataset.py in Cloud Standalone Mode...")
        at_dataset = AppTest.from_file("app/pages/dataset.py")
        at_dataset.run(timeout=20)
        if at_dataset.exception:
            for ex in at_dataset.exception:
                print("ERROR in standalone dataset page:", ex.message)
            raise RuntimeError("Cloud Standalone mode failed on dataset.py!")
        print("✅ Dataset page successfully rendered in Cloud Standalone Mode!")

        # Test Results page in standalone mode
        at_results = AppTest.from_file("app/pages/results.py")
        at_results.run(timeout=20)
        assert not at_results.exception, f"Results standalone exception: {at_results.exception}"
        print("✅ Results page successfully rendered in Cloud Standalone Mode!")

    finally:
        # Restore raw CSV
        print(f"Restoring {BACKUP_PATH} -> {DATA_PATH} ...")
        shutil.move(str(BACKUP_PATH), str(DATA_PATH))
        assert DATA_PATH.exists(), "Failed to restore DATA_PATH!"
        assert DATA_PATH.stat().st_size == original_size, "File size mismatch after restore!"
        print(f"✅ Successfully restored dataset ({original_size:,} bytes) intact.")


def main():
    print("=" * 60)
    print("STARTING STREAMLIT VALIDATION SUITE")
    print("=" * 60)

    # 1. Test all 5 pages
    test_page("app/pages/home.py", "Page 1: Home")
    test_page("app/pages/dataset.py", "Page 2: Dataset Overview")
    test_page("app/pages/feature_selection.py", "Page 3: Feature Selection")
    test_page("app/pages/loss_functions.py", "Page 4: Loss Functions")
    test_page("app/pages/results.py", "Page 5: Results & Conclusions")

    # 2. Test interactive metric toggles
    test_results_page_interactions()

    # 3. Test Cloud Standalone mode
    test_cloud_standalone_mode()

    print("\n" + "=" * 60)
    print(">>> ALL 5 PAGES & STANDALONE TESTS PASSED WITH ZERO ERRORS <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
