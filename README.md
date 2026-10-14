# Human Activity Recognition: Walking vs. Jumping

A data science project that classifies phone accelerometer recordings as **walking** or **jumping**. Three people (Dayna, Ashley and Nick) each recorded both activities, and the script cleans the data, extracts features, trains a logistic regression classifier, and includes a small desktop app for classifying a new recording.

## Data

Each recording is a CSV of phone accelerometer readings sampled at about 100 Hz, with columns:

`Time (s)`, `Acceleration x (m/s^2)`, `Acceleration y (m/s^2)`, `Acceleration z (m/s^2)`, `Absolute acceleration (m/s^2)`

| Activity | Dayna | Ashley | Nick |
| --- | --- | --- | --- |
| Walking | `dayna-walk.csv` | `walking-Ashley.csv` | `nickWalk.csv` |
| Jumping | `dayna-jump.csv` | `jumping-Ashley.csv` | `nickJump.csv` |

- `Meta Data/` holds each person's device details and the start/pause timestamps of each recording.
- `test.csv`, `test2.csv` and `test3.csv` are extra recordings for trying the classifier.
- `motion_data.h5` is the segmented dataset the script writes out (walking and jumping segments).

## Pipeline (`DataSciCode.py`)

1. **Load and label** the six recordings.
2. **Segment** each recording into 5-second windows (500 samples).
3. **Store** the segmented data in `motion_data.h5`.
4. **Visualize** raw signals for each person.
5. **Clean** with a rolling-mean filter (window of 5 samples) and re-segment.
6. **Extract features** per axis: max, min, range, mean, standard deviation, median, skewness, kurtosis, variance and RMS.
7. **Normalize** the features with `StandardScaler`.
8. **Train** a `LogisticRegression` classifier and report accuracy and recall, with a confusion matrix and ROC curve.
9. **Desktop app (Tkinter):** pick a CSV, get a walking/jumping prediction for each 5-second segment, view the results as a bar chart and scatter plot, and save the predictions to a CSV.

## Running it

```bash
pip install pandas numpy scipy scikit-learn matplotlib h5py
python DataSciCode.py
```

Run it from this folder, since the CSV files are loaded by relative path. Tkinter ships with most Python installs (on Linux you may need `sudo apt install python3-tk`).

## Notes

In the current script the training and test feature sets are both built from the same cleaned segments (`clean_segments`), so the reported metrics describe performance on the training data rather than on held-out data. Using the `train_test_split` that is already in the script to build separate train and test features would give a fair estimate.
