#!/usr/bin/env bash

kaggle competitions download -c predicting-red-hat-business-value -p .
unzip predicting-red-hat-business-value.zip -d tmp
unzip tmp/*.csv.zip -d packages/lr_customer_value/lr_customer_value/datasets/
