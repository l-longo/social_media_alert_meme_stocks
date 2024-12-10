# A Social Media Alert System for Meme Stocks

*We plan to update the data and the model's results for 2024 soon, and to provide regular updates from 01-2025 onwards.*

This document explains how to reproduce results in the paper: **A Social Media Alert System for Meme Stocks**.

The model consists in an alert system designed to detect potential unusual activity in terms of user discussions on certain securities' tickers traded in the financial markets.
To achieve the results we extract network dimensions from Reddit, and we convert them in potential signals to predict future abnormal returns. 

<img src="figures/Network_creation.PNG" alt="Reddit social structure" width="600">

<br><br>

**Data decompression and processing**: 
The first step is the decompression of the zst files containing the raw Reddit data. To download the data, we rely on the project built by https://github.com/Watchful1/PushshiftDumps. You can find the wallstreetbets data at this open-access drive folder: https://drive.google.com/drive/folders/1Y6lpnhT5mXh5q-D_xLDpF9tzWoRMmJXl?usp=sharing.
The data are in zst format: use the scipt *decompress_zst.py* to extract the raw data in a csv format, then *open.py* to merge the comments and submissions. The following graph shows the network of users interacting on Reddit on January 14, 2021:

<img src="figures/GME_network_graph_14_01_2021_lighter_background.png" alt="Reddit social structure" width="400">

<br><br>

**Training/Test of the model**:
As explained in Section 5.2 of the paper, the alert system is trained on data up to 2022m5. The script *train.py* reproduces the training exercise with a grid-search algorithm. The script *test.py* reproduces the out-of-sample evaluation of the trained model.

