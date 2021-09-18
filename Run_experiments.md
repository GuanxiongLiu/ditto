Ditto

Edit file: run_extend_demo.sh
- dataset=femnist
- Change: boosting, random_updates, gradient_clipping

Edit file: main_extend.py
- Ditto = DittoOpt
- Meta-learning = MetaOpt
- Private layer = PrivateLayerOpt
- Local SGD Opt = LocalOpt
- Multi Krum = MultiKrumAgg
- Elementwise Median = ElementWiseMedian
- Elementwise Trimmed Mean = ElementWiseTrimmedMean
- Simple FedAvg = SimpleFedAvg

Setup local optimizer:
- t.set_local_optimizer(DittoOpt…
Setup local aggregator:
- t.set_global_aggregator(ElementWiseMedian…

Run the experiments with command:
nohup ./run_extend_demo.sh extend &> <path to customized log file> &
