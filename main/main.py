import wfdb


def temp():
    ecg_record = wfdb.rdrecord('mit-bih/108')
    wfdb.plot_wfdb(ecg_record)
    pass


def main():
    temp()

    pass


if __name__ == "__main__":
    main()
