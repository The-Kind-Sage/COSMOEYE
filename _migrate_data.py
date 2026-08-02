"""Move data from legacy datasets/ into data/. Delete this script after running."""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD  = os.path.join(ROOT, "datasets")
NEW  = os.path.join(ROOT, "data")


def move_tree(src, dst):
    """Move all contents of src into dst, merging if dst exists."""
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            move_tree(s, d)
            try:
                os.rmdir(s)
            except OSError:
                pass
        else:
            if os.path.exists(d):
                print(f"  SKIP (exists): {d}")
            else:
                shutil.move(s, d)
                print(f"  MOVED  {s}")


if not os.path.isdir(OLD):
    print(f"Nothing to move – {OLD} does not exist.")
else:
    # datasets/s2/   -> data/raw/s2/
    s2_old = os.path.join(OLD, "s2")
    s2_new = os.path.join(NEW, "raw", "s2")
    if os.path.isdir(s2_old):
        print(f"\nMoving s2/ raw tiles -> data/raw/s2/ ...")
        move_tree(s2_old, s2_new)

    # datasets/TrainData/ -> data/TrainData/
    td_old = os.path.join(OLD, "TrainData")
    td_new = os.path.join(NEW, "TrainData")
    if os.path.isdir(td_old):
        print(f"\nMoving TrainData/ -> data/TrainData/ ...")
        move_tree(td_old, td_new)

    # datasets/TestData/ -> data/TestData/
    test_old = os.path.join(OLD, "TestData")
    test_new = os.path.join(NEW, "TestData")
    if os.path.isdir(test_old):
        print(f"\nMoving TestData/ -> data/TestData/ ...")
        move_tree(test_old, test_new)

    # Remove the now-empty datasets/ root
    try:
        os.rmdir(OLD)
        print(f"\nRemoved empty: {OLD}")
    except OSError as e:
        print(f"\nCould not remove {OLD}: {e}")
        print("  (it may still contain files — check manually)")

print("\nData migration complete.")
