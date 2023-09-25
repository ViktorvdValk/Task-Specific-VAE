from utils.helper_data_split import (
    run_splits,
    train_test_val_split,
    run_just_scale,
    run_withings,
    run_small_data_set2,
    create_small_data_set,
)


if __name__ == "__main__":
    # run_withings(nr_splits=3)
    # run_just_scale()
    run_splits(3, save=True, test=True)
    # run_small_data_set2()