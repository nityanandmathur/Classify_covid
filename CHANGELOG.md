# Changelog

## Version 0.1 (development)

1. Added pip dependencies to tensorflow env file - dvc[all], wandb, hydra-core, scikit-learn, __pillow__(Necessary for this model. not installing pillow throws error.)

2. `wandb login`

3. `dvc init`

4. Loaded training data to _data/raw_ folder.

5. Remove "" from config file.

5. Removed _data_ from _git_. 

```bash
git rm -r --cached 'data'
```

> Trying to add to DVC versioning takes long time on docker container on local machine. Suggested to do on server only.

6. In file __train_model.py__
    1. Added configuration to WandB - `wandb.init(project="classify_covid", config=cfg)`
    2. Added main function with arguments parser

    ```python
    
    if __name__ == "__main__":
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--config', dest='config', required=True)
        args = argparser.parse_args()

        train(config_name=args.config)
    ```

    3. Converted the file structure to a function `train(config_name: Text) -> None:`

    4. Changed `conf/` to `configs/`

    5. Changed loading of _c, n, p_ from direct paths to path from config file. 

    6. Added `model.save()` to save model post training.

    7. Changed `lr = 0.01` to `lr = cfg.params.learning_rate` 

7. Added model save location to _config.yaml_

8. Added stage in dvc for training model.

```bash
dvc stage add -n train -p configs/config.yaml -o models/covid.h5 scripts/train_model.py --config=config.yaml
# not added -d data/raw, since data is not cached. For training in server add: -d data/raw
```
9. Added `model.py` & `preprocessing.py` to _dvc.yaml: deps_
