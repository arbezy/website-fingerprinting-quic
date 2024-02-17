# parse csv into traffic features


# unique pkt size, packet size count, packer order, 
# inter-arrival time (should i do this one?), negative packets (number of traffic going from server to client)
# cumulative size, cumulative size with direction, burst number/maximal length/mean length,
# total transmission time


# also need to classify traffic into positive/negative, tiny/small/medium/large

# need to remove handshake data as it is done in the early quic paper


import pandas as pd
import numpy as np
import os
import sys

# TODO: need to write results of feature extraction somewhere... another dataframe that I can export to csv?

def main():
    for f in os.listdir(sys.argv[1]):
        current_capture = pd.read_csv(f"{f}")
        simple_features(current_capture)
        
def remove_quic_handshake():
    pass

def remove_tcp_handshake():
    pass

def label_dataframe(capture_df: pd.DataFrame) -> pd.DataFrame:
    pass
        
        
# SIMPLE FEATURES:

def simple_features(capture_df: pd.DataFrame):
    simple_features_results = {s1+s2:0 for s1 in ["positive", "negative"] for s2 in ["tiny", "small", "medium", "large"]}
    for index, row in capture_df.iterrows():
        # TODO: find out which col src and dst point are in after parsing
        src_port = row
        dst_port = row
        pkt_size = row
        if src_port == 2020 and dst_port == 58762:
            simple_features_results["positive"+get_pkt_size_classification(pkt_size)]
        elif src_port == 2020 and dst_port == 58762:
           simple_features_results["negative"+get_pkt_size_classification(pkt_size)]
        else:
            print("Neither the src or dst ip matched the client or server IP")
    # TODO: may want to perform some transformation on this dictionary to make it easier for pandas to parse :)
    return simple_features_results
        
        
def get_pkt_size_classification(pkt_size: bytes) -> str:
    # unit == bytes
    # TODO: write critique of paper that they used less than when they meant less than or equal to
    if pkt_size < 80:
        return "tiny"
    elif 80 <= pkt_size and pkt_size < 160:
        return "small"
    elif 160 <= pkt_size and pkt_size < 1280:
        return "medium"
    elif 1280 <= pkt_size:
        return "large"
    else:
        raise Exception("hmmm invalid pkt size, that's weird")



# ADVANCED FEATURES:

"""this feature statistics whether the packet 𝑡 of
length 𝑙 is in the traffic 𝑇 . Specifically, define Length(𝑡) as a func-
tion that calculates the length of packet 𝑡, if 𝑙 ∈ {Length(𝑡𝑖)|𝑡𝑖 ∈
𝑇 }, 𝑙-th dimension of this feature is set to 1, otherwise, is set to
0. This feature is a 1460-dimension vector (packet length range
from 54 to 1514).
"""
# first i need to know the range of my packet sizes!
def unique_packet_size(capture_df: pd.DataFrame):
    # using the length of the data rather than the length of the packet as I think it makes more sense
    # TODO: check this over with Marc
    # TODO: change these values to match my actual data
    smallest_pkt_size = 54
    largest_pkt_size = 1514
    unique_packet_sizes = [0 for i in range(largest_pkt_size - smallest_pkt_size + 1)]
    for index, row in capture_df.iterrows():
        pkt_size = row['data_length']
        unique_packet_sizes[pkt_size-54] = 1
    # TODO: return a pd / np array?
    return unique_packet_sizes
    
        
"""packet size count: this feature statistics the number of packet 𝑡 of
length 𝑙 in the traffic 𝑇 . Specifically, if 𝑙 ∈ {Length(𝑡𝑖)|𝑡𝑖 ∈ 𝑇 }, 𝑙-th
dimension of this feature is set to Card({𝑡𝑖|𝑡𝑖 ∈ 𝑇 , Length(𝑡𝑖) = 𝑙}).
This feature is a 1460-dimension vector.
"""
def pkt_size_count(capture_df: pd.DataFrame):
    # TODO: change pkt size range
    smallest_pkt_size = 54
    largest_pkt_size = 1514
    unique_packet_sizes = [0 for i in range(largest_pkt_size - smallest_pkt_size + 1)]
    for index, row in capture_df.iterrows():
        pkt_size = row['data_length']
        unique_packet_sizes[pkt_size-54] += 1
    return unique_packet_sizes

"""packet order: this feature records the packets length in order
of packet position. Specifically, the 𝑖-th dimension of this feature
is set to Length(𝑡𝑖), where 𝑡𝑖 is the 𝑖-th packet in traffic 𝑇 . This
feature is a 𝑘-dimension vector.
"""
def packet_order(capture_df):
    return capture_df['pkt_length']

"""inter-arrival time: this feature statistics arrival interval of ad-
jacent packets in order of packet position. Specifically, define
Time(𝑡) as a function that fetch the arrival time of a packet, let
𝑡0 be the 2-nd Client Hello packet for GQUIC, the last Handshake
packet for IQUIC, and the Change Cipher Spec packet for HTTPS,
then 𝑙-th dimension of this feature is set to (Time(𝑡𝑖) − Time(𝑡𝑖−1)),
where 𝑡𝑖, 𝑡𝑖−1 ∈ 𝑇 . This feature is a 𝑘-dimension vector.
"""
def interarrival_time():
    pass

"""negative packets: this feature statistics the number of packet 𝑡
in negative direction in traffic 𝑇 . This 1-dimension feature is set
to Card({(𝑡𝑗 , 𝑑𝑗 )|𝑑𝑗 = negative}).
"""
def neg_pkts():
    pass

"""cumulative size: this feature statistics the cumulative size of
packets in traffic 𝑇 . This 1-dimension feature is set to ∑{𝑇𝑝, 𝑇𝑛},
where 𝑇𝑝 = {Length((𝑡𝑖, 𝑑𝑖))|𝑡𝑖 ∈ 𝑇 , 𝑑𝑖 = positive}, 𝑇𝑛 =
{Length((𝑡𝑖, 𝑑𝑖))|𝑡𝑖 ∈ 𝑇 , 𝑑𝑖 = negative}.

THIS IS SUPER MISLEADING IN THE PAPER, CUMULATIVE SIZE IS JUST TOTAL SIZE AS IT IS 1 DIMENSION
"""
def cumulative_size():
    pass

def
        
            
    
    
main()