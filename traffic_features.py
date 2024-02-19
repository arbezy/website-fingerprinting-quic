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
        current_capture = pd.read_csv(f"{f}", sep='\t')
        
        if is_quic:
            current_capture = label_quic_dataframe(current_capture)
            current_capture = filter_out_irrelevant_pkts_quic(current_capture)
            current_capture = remove_quic_handshake(current_capture)
        else:
            current_capture = label_tcp_dataframe(current_capture)
            current_capture = filter_out_irrelevant_pkts_tcp(current_capture)
            current_capture = remove_tcp_handshake(current_capture)
            
        simple = simple_features(current_capture)
        transfer = transfer_features(current_capture)
        
        
def filter_out_irrelevant_pkts_quic(df: pd.DataFrame) -> pd.DataFrame:
    # checking source ports or ip addresses
    df1 = df[df['dst_port'] == 2020]
    df2 = df[df['src_port'] == 2020]
    df1 = df1.append(df2, ignore_index=False)
    df1 = df1.sort_values(by=['seq_num'])
    return df1

def filter_out_irrelevant_pkts_tcp(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df[df['dst_port_tcp'] == 2020]
    df2 = df[df['src_port_tcp'] == 2020]
    df3 = df[df['proto'] != 17]
    df1 = df1.append(df2, ignore_index=False)
    df1 = df1.append(df3, ignore_index=False)
    df1 = df1.sort_values(by=['seq_num'])
    return df1
        
def remove_quic_handshake():
    # TODO: may need to parse the traffic differently to figure this one out
    pass

def remove_tcp_handshake():
    
    pass

def label_quic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # data_length, pkt_length, arrival_time
    df = df.assign(e=pd.Series(range(1, len(df)+1)).values)
    df.columns = ["time_frame_epoch", "src_ip", "dst_ip", "src_port", "dst_port", "src_port_tcp", "dst_port_tcp", "proto", "ip_len", "ip_hdr_len", "data_len", "udp_len", "time_delta", "time_relative", "udp_stream", "expert_msg", "seq_num"]
    return df
        
# TODO: need to change this to the correct columns
def label_tcp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(e=pd.Series(range(1, len(df)+1)).values)
    df.columns = ["time_frame_epoch", "src_ip", "dst_ip", "src_port", "dst_port", "src_port_tcp", "dst_port_tcp", "proto", "ip_len", "ip_hdr_len", "data_len", "udp_len", "time_delta", "time_relative", "udp_stream", "expert_msg", "seq_num"]
    return df
    
# SIMPLE FEATURES:

def simple_features(capture_df: pd.DataFrame):
    simple_features_results = {s1+s2:0 for s1 in ["positive", "negative"] for s2 in ["tiny", "small", "medium", "large"]}
    for index, row in capture_df.iterrows():
        # TODO: find out which col src and dst point are in after parsing
        src_port = row.src_port
        dst_port = row.dst_port
        pkt_size = row.data_len
        # server port is 2020
        if src_port == 2020:
            simple_features_results["negative"+get_pkt_size_classification(pkt_size)]
        elif dst_port == 2020:
           simple_features_results["positive"+get_pkt_size_classification(pkt_size)]
        else:
            print("Neither the src or dst ip matched the client or server IP (but that's ok)")
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

def transfer_features():
    pass

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
def interarrival_time(capture_df):
    interarrival_times = [-1 for i in range(len(capture_df))]
    previous_row_atime = 0
    
    for index, row in capture_df.iterrows():
        interarrival_times.append(row['arrival_time'] - previous_row_atime)
        previous_row_atime = row['arrival_time']
        
    return interarrival_times

"""negative packets: this feature statistics the number of packet 𝑡
in negative direction in traffic 𝑇 . This 1-dimension feature is set
to Card({(𝑡𝑗 , 𝑑𝑗 )|𝑑𝑗 = negative}).
"""
def neg_pkts(simple_features):
    return (simple_features["negativetiny"] + simple_features["negativesmall"] + simple_features["negativemedium"] + simple_features["negativelarge"])

"""cumulative size: this feature statistics the cumulative size of
packets in traffic 𝑇 . This 1-dimension feature is set to ∑{𝑇𝑝, 𝑇𝑛},
where 𝑇𝑝 = {Length((𝑡𝑖, 𝑑𝑖))|𝑡𝑖 ∈ 𝑇 , 𝑑𝑖 = positive}, 𝑇𝑛 =
{Length((𝑡𝑖, 𝑑𝑖))|𝑡𝑖 ∈ 𝑇 , 𝑑𝑖 = negative}.

THIS IS SUPER MISLEADING IN THE PAPER, CUMULATIVE SIZE IS JUST TOTAL SIZE AS IT IS 1 DIMENSION
"""
def cumulative_size(capture_df: pd.DataFrame):
    return capture_df['pkt_length'].cumsum()

"""cumulative size with direction: this feature statistics the cu-
mulative size of packets in traffic 𝑇 , but the impact of packet
direction 𝑑 is considered. This 1-dimension feature is set to
∑{𝑇𝑝, 𝑇𝑛}, where 𝑇𝑝 = {Length((𝑡𝑖, 𝑑𝑖))|𝑡𝑖 ∈ 𝑇 , 𝑑𝑖 = positive}, 𝑇𝑛 =
{−Length((𝑡𝑖, 𝑑𝑖))|𝑡𝑖 ∈ 𝑇 , 𝑑𝑖 = negative}.
"""
def cumulative_size_w_direction(capture_df: pd.DataFrame):
    cumulative_sum = 0
    for index, row in capture_df.iterrows():
        src_port = row.src_port
        dst_port = row.dst_port
        if src_port == 2020 and dst_port == 58762:
            cumulative_sum += row.pkt_length
        elif dst_port == 2020 and src_port == 58762:
            cumulative_sum -= row.pkt_length
    return cumulative_sum

"""bursts numbers/maximal length/mean length: burst is define
as the consecutive packets between two packets sent in the oppo-
site direction [33]. Bursts numbers, bursts maximal length, and
bursts mean length is the statistical features based on burst in the
traffic 𝑇
"""
def burst_features():
    pass

"""total transmission time: this feature statistics the total trans-
mission time of traffic 𝑇 . This 1-dimension feature is set to
∑{Time(𝑡𝑖) − Time(𝑡𝑖−1)|𝑡𝑖 ∈ 𝑇 , 𝑖 > 1}
"""
def total_transmission_time(df: pd.DataFrame):
    # think here I can just get the time stamp of the final packet
    final_elem = df.iloc(-1)
    return final_elem.time_relative





is_quic = False
if sys.argv[2] == "quic":
    is_quic = True  
main()