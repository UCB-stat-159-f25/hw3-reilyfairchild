from ligotools import readligo as rl
import numpy as np


def test_segment_creation():
    """
    Tests the dq_channel_to_seglist function.
    This function turns a 1D array of 0s and 1s into a list of slices.
    """
    print("Running test_segment_creation...")
    
    # A simple channel with one segment of 1s
    # Indices: 0  1  2  3  4  5
    channel = np.array([0, 0, 1, 1, 1, 0])
    
    # We expect one segment (slice) from index 2 up to (but not including) 5
    expected_slices = [slice(2, 5, None)]
    
    # Run the function with a sample frequency (fs) of 1
    result_slices = rl.dq_channel_to_seglist(channel, fs=1)
    
    assert result_slices == expected_slices, f"Expected {expected_slices}, but got {result_slices}"
    
    # --- A second check with a different sample rate ---
    
    # We expect the same indices, but multiplied by the sample rate
    expected_slices_fs4 = [slice(2 * 4, 5 * 4, None)] # slice(8, 20, None)
    
    result_slices_fs4 = rl.dq_channel_to_seglist(channel, fs=4)
    
    assert result_slices_fs4 == expected_slices_fs4, f"Expected {expected_slices_fs4}, but got {result_slices_fs4}"

    print("...test_segment_creation PASSED")


def test_gps_segment_conversion():
    """
    Tests the dq2segs function.
    This function turns a 1D array into a SegmentList of (start_gps, stop_gps) tuples.
    """
    print("\nRunning test_gps_segment_conversion...")

    # A channel with two segments
    # Indices: 0  1  2  3  4  5  6  7
    channel = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    
    # The GPS start time of this channel
    gps_start = 1000
    
    # Expected segments:
    # 1. (1000 + index 1) to (1000 + index 3) -> (1001, 1003)
    # 2. (1000 + index 4) to (1000 + index 7) -> (1004, 1007)
    expected_seglist = [(1001, 1003), (1004, 1007)]
    
    # Run the function
    result_segment_list_obj = rl.dq2segs(channel, gps_start)
    
    # Get the list of segments from the object
    result_seglist = result_segment_list_obj.seglist
    
    assert result_seglist == expected_seglist, f"Expected {expected_seglist}, but got {result_seglist}"

    print("...test_gps_segment_conversion PASSED")


if __name__ == "__main__":
    print("--- Starting Simple Tests ---")
    test_segment_creation()
    test_gps_segment_conversion()
    print("\n--- All Simple Tests Passed! ---")