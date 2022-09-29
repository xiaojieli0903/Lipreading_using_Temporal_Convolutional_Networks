flake8 . --exclude train_logs
isort -rc .
yapf -r -i .
