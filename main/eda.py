import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_models(mode):
    folder_path = f"../log/pipeline/{mode}"
    models = ["LeNet5", "AlexNet", "GoogLeNet", "ResNet34", "VGGNetD"]

    for folder_name in models:
        folder = os.path.join(folder_path, folder_name)
        if not os.path.exists(folder) or not os.path.isdir(folder):
            models.remove(folder_name)
    return models


def get_values(data_dict):
    accuracy_values = []
    val_accuracy_values = []

    for key, value in data_dict.items():
        if "val_dense" in key and "accuracy" in key:
            val_accuracy_values.append(value)
        elif "dense" in key and "accuracy" in key:
            accuracy_values.append(value)

    accuracy = max(accuracy_values)
    val_accuracy = max(val_accuracy_values)
    return accuracy, val_accuracy


def get_head_data(log_data, head_delim):
    headers = log_data.iloc[:head_delim, 0].tolist()
    data = log_data.iloc[head_delim:, 0].tolist()
    return headers, data


def get_data_from_logs(main_frame, headers, iteration, filename, model):
    headers.insert(0, "validation_split")
    headers.insert(0, "batch_size")
    headers.insert(0, "epochs")
    headers.insert(0, "model")
    # for i in range(len(iteration)):
    iteration[-1].insert(0, filename[2])
    iteration[-1].insert(0, filename[1])
    iteration[-1].insert(0, filename[0])
    iteration[-1].insert(0, model)
    df_iteration = pd.DataFrame([iteration[-1]], columns=headers)
    main_frame = pd.concat([main_frame, df_iteration], ignore_index=True)
    return main_frame


def read_logs(mode):
    models = check_models(mode)
    main_frame = pd.DataFrame()
    for model in models:
        log_folder = os.path.join("../log", "pipeline", mode, model)

        log_files = [file for file in os.listdir(log_folder) if file.endswith(".csv")]
        for log_file in log_files:
            filename = os.path.splitext(log_file)[0].split('-')
            log_data = pd.read_csv(os.path.join(log_folder, log_file), header=None)
            if mode == "train":
                iterations = list()
                if model == "GoogLeNet":
                    head_delim = 15
                    headers, data = get_head_data(log_data, head_delim)
                    log_dict = dict()
                    for i in range(len(data) // len(headers)):
                        for j in range(0, head_delim):
                            log_dict.setdefault(headers[j], []).append(data[j + (head_delim * i)])
                        temp = {key: value[i] for key, value in log_dict.items()}
                        accuracy_avg, val_accuracy_avg = \
                            get_values(temp)
                        iterations.append([
                            data[0 + (i * head_delim)],
                            accuracy_avg,
                            data[7 + (i * head_delim)],
                            val_accuracy_avg,
                            data[14 + (i * head_delim)]
                        ])
                    headers = list([
                        "epoch",
                        "accuracy",
                        "loss",
                        "val_accuracy",
                        "val_loss"
                    ])
                else:
                    head_delim = 5
                    headers, data = get_head_data(log_data, head_delim)
                    for i in range(len(data) // len(headers)):
                        iterations.append(data[head_delim * i:head_delim * (i + 1)])
                    # last_iter = iterations[-1]
                main_frame = get_data_from_logs(main_frame, headers, iterations, filename, model)
            elif mode == "test":
                head_delim = 2
                headers, data = get_head_data(log_data, head_delim)
                iterations = list()
                if model == "GoogLeNet":
                    data_delim = 7
                    for i in range(len(data) // data_delim):
                        loss = data[0 + (i * data_delim)]
                        acc = (data[4 + (i * data_delim):(i + 1) * data_delim])
                        accuracy = max(acc)
                        iterations.append([
                            loss,
                            accuracy
                        ])
                else:
                    for i in range(len(data) // len(headers)):
                        iterations.append(data[head_delim * i:head_delim * (i + 1)])
                    # last_iter = iterations[-1]
                main_frame = get_data_from_logs(main_frame, headers, iterations, filename, model)
    if mode == "train":
        return main_frame.iloc[:, :-2]
    return main_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Generating result data table")
    parser.add_argument("-m", help="Train or test", required=True)
    parser.add_argument("-op", help="Just plot", required=True)
    return parser.parse_args()


def generate_plot(data_path, mode):
    data = pd.read_csv(data_path)

    model_params = data[['model', 'epochs', 'batch_size', 'validation_split']]
    accuracy = data['accuracy']

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_params.apply(lambda x: f"{x['model']}\n"
                                               f"{x['epochs']},"
                                               f"{x['batch_size']},"
                                               f"{x['validation_split']}",
                                     axis=1), y=accuracy)
    plt.xlabel("Model with params")
    plt.ylabel("Accuracy")
    plt.title("Accuracy/models")
    plt.xticks(rotation=90)
    plt.savefig(f"../result/{mode}/plot.png")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    mode = args.m
    just_plot = args.op
    res_path = f"../result/{mode}"
    if not just_plot:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        log_data = read_logs(mode)
        os.makedirs(res_path, exist_ok=True)
        log_data.to_csv(res_path + "/data.csv", index=False)
        log_data = pd.read_excel(res_path + "/data.xlsx")
        log_data.to_csv(res_path + "/data.csv", index=False)
        if mode == "train":
            print(f"Train result data has been generated into file:\n{res_path}/data.xlsx")
        elif mode == "test":
            print(f"Test result data has been generated into file:\n{res_path}/data.xlsx")
    file_path = res_path + "/data.csv"
    generate_plot(file_path, mode)


if __name__ == "__main__":
    main()
