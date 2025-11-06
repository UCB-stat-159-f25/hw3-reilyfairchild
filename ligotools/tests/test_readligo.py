from ligotools import readligo as rl
import numpy as np
import json


def test_loaddata():
    eventname = 'GW150914' 
    events = json.load(open("data/BBH_events_v3.json","r"))
    event = events[eventname]
    fn_H1 = 'data/'+event['fn_H1'] 

    try:
        # read in data from H1 and L1, if available:
        strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
        #strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
    except:
        print("Cannot find data files!")
        print("You can download them from https://www.gwosc.org/s/events/"+eventname)
        print("Quitting.")
        quit()
        
    assert strain_H1.shape == (131072,)
    print("...test_loaddata() PASSED")


def test_dq():
    eventname = 'GW150914' 
    events = json.load(open("data/BBH_events_v3.json","r"))
    event = events[eventname]
    fn_L1 = 'data/'+event['fn_L1']  
    strain, time, chan_dict = rl.loaddata(fn_L1, 'H1')
    DQflag = 'CBC_CAT3'
    # readligo.py method for computing segments (start and stop times with continuous valid data):
    segment_list = rl.dq_channel_to_seglist(chan_dict[DQflag])
    assert len(segment_list) == 1
    print("...test_loaddata() PASSED")

if __name__ == "__main__":
    print("--- Starting Simple Tests ---")
    
    test_loaddata()
    test_dq()
    print("\n--- All Simple Tests Passed! ---")