Configuration

Edit file: run_extend_demo.sh
* dataset=<the dataset that you want to run experiments, e.g., femnist> 
* attack behavior:
    * behavior #1: boosting = 0, random_updates = 0
    * behavior #2: boosting = 0, random_updates = 1
    * behavior #3: boosting = 1, random_updates = 0
* gradient_clipping=<0 or 1>

Edit file: main_extend.py
* Local updating:
    * Ditto = DittoOpt
    * Meta learning = MetaOpt
    * Private layer = PrivateLayerOpt
    * Vanilla SGD update (no personalization) = LocalOpt
* Global aggregation:
    * Multi Krum = MultiKrumAgg
    * Elementwise Median = ElementWiseMedian
    * Elementwise Trimmed Mean = ElementWiseTrimmedMean
    * Simple FedAvg (no defense) = SimpleFedAvg

Setup local optimizer:
* t.set_local_optimizer(DittoOpt…
Setup local aggregator:
* t.set_global_aggregator(ElementWiseMedian…

Run the experiments with command:
nohup ./run_extend_demo.sh extend &> <path to customized log file> &
