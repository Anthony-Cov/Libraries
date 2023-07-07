# Libraries
<H2>Some procedures and utilities for time series forecast and analysis which were included in some other projects.</H2>
<H3>Forecasting</H3>
<LI><I>Autoregr.py</I> - forecast model of Vector Autoregression with additional series.
<LI><I>ChooChoo.py</I> - forecast model of Maximal Similarity with additional series.(described in A. Kovantsev, P. Chunaev and K. Bochenina, "Evaluating Time Series Predictability via Transition Graph Analysis," 2021 International Conference on Data Mining Workshops (ICDMW), Auckland, New Zealand, 2021, pp. 1039-1046, <A href=https://ieeexplore.ieee.org/document/9679876>doi: 10.1109/ICDMW53433.2021.00135</A> section III-B)   
<LI><I>Localapp.py</I> - forecast model of Local Approximation with additional series. (described in A. Kovantsev and P. Gladilin, "Analysis of multivariate time series predictability based on their features," 2020 International Conference on Data Mining Workshops (ICDMW), Sorrento, Italy, 2020, pp. 348-355 <A href=https://ieeexplore.ieee.org/document/9346469)>doi: 10.1109/ICDMW51313.2020.00055</A> section V-B)
<LI><I>NeurosV.py</I> - forecast model of LSTM neural network with additional series.
<LI><I>Spectrum.py</I> - forecast model of MSSA with additional series.<BR>
for tests: <I>Autoregr.VARExplore, Spectrum.MSSAExplore, NeurosV.LSTMExploreV, Localapp.LAprExplore ChooChoo.ChooChooExplore</I>. The input parameters of the procedures are also unified: a time series of data for one of the regions, a list of predictor dictionaries obtained from selection procedures or an empty list when working without predictors, the forecast horizon, which is also the size of the test sample, the split point is the position of the last value of the training sample from the end original row. The output of all procedures is a tuple containing the values of the mean error, mean absolute error, mean relative percentage error, symmetric mean relative percent error, standard deviation, and predicted series for the test subset of values. All errors are calculated for test data;
<H3>Time series features used in the research</H3>
<LI><I>EmbDim.cpp</I> - dynamic system embedding dimension used for the Maximal Similarity and Local Approximation methods. Should be compiled with GCC like <I>"gcc -fPIC -shared -o EmbDim.so EmbDim.cpp -lstdc++"</I> to get EmbDim.so
<LI><I>EmbDim.h</I> - C++ header for compillation
<LI><I>EmbDim.so</I> - object module for python implementation
<LI><I>features.py</I> - contains the procedure of embedding dimension calculation with no C++, which works much slower
<H3>Causality tests for predictors choice</H3>
<LI><I>Tests.py</I> - includes cross-correlation, Granger's test, Convergent Cross Mapping (CCM)<BR>
For choosing predictors: <I>Tests.ChoosePredsGran</I>I> - by Granger test, <I>Tests.ChoosePredsCCor</I> - by cross-correlation, <I>Tests.ChoosePredsCCM</I> - by convergent joint representation, <I>Tests.ChoosePredsVAR</I> - by vector autoregression forecast quality. The input parameters of the procedures are unified: a time series of data for one of the regions, a list of names of available predictor files, the number of predictors for selection. The output is a list of dictionaries containing the file name, the value of the selection criterion, the time lag at which this criterion takes the best value;
<H3>Utilites</H3>
<LI><I>Util.py</I> - utilites like series rescaling, quality metrics calculation, noise filtering and so on
<H3>Idle in this research</H3>  
<LI><I>CorrEntr.cpp</I>, <I>CorrEntr.h</I>, <I>CorrEntr.so</I> - Correlation entropy
<LI><I>fakeseries.py</I> and <I>generator.py</I> - artificial time series generator
<LI><I>graph.py</I> - time series to transition grapg transformation
<LI><I>HurstExp.cpp</I>, <I>HurstExp.h</I>, <I>HurstExp.so</I>- Hurst exponent calculation
