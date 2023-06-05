import os
import wfdb
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from tqdm import tqdm


# centering around peak
mode = 128
output_dir = "../data"

image_size = 128

fig = plt.figure()
dpi = fig.dpi

fig_size = (image_size / dpi, image_size / dpi)
image_size = (image_size, image_size)


def plot(signal, filename):
    plt.figure(figsize=fig_size, frameon=False)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.plot(signal)
    plt.savefig(filename)

    plt.close()

    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.resize(im_gray, image_size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(filename, im_gray)


def data_gen(ecg):
    name = os.path.basename(ecg)
    record = wfdb.rdrecord(ecg)
    ann = wfdb.rdann(ecg, extension="atr")
    for sig_name, signal in zip(record.sig_name, record.p_signal.T):
        # Check for NaN in signal
        if not np.all(np.isfinite(signal)):
            continue
        signal = scale(signal)
        for i, (label, peak) in tqdm(enumerate(zip(ann.symbol, ann.sample)), total=len(ann.sample), desc="Peaks read"):
            if label == '/':
                label = "slash"
            elif label == '|':
                label = "pipe"
            elif label == '*':
                label = "asterisk"
            elif label == "\"":
                label = "quote"
            # print(f"{name} {sig_name} [{i + 1}/{len(ann.symbol)}]", end='\r')
            if isinstance(mode, int):
                left, right = peak - mode // 2, peak + mode // 2
            elif isinstance(mode, list):
                if (i > 0) and ((i + 1) < len(ann.sample)):
                    left = ann.sample[i - 1] + mode[0]
                    right = ann.sample[i + 1] + mode[1]
                else:
                    continue
            else:
                raise Exception("Incorrect mode value!")
            if (left > 0) and (right < len(signal)):
                d1_data_dir = os.path.join(output_dir, "1D", name, sig_name, label)
                d2_data_dir = os.path.join(output_dir, "2D", name, sig_name, label)
                os.makedirs(d1_data_dir, exist_ok=True)
                os.makedirs(d2_data_dir, exist_ok=True)

                filename = os.path.join(d1_data_dir, f"{peak}.npy")
                np.save(filename, signal[left:right])
                filename = os.path.join(d2_data_dir, f"{peak}.png")
                plot(signal[left:right], filename)

    print(f"Data generation of {name} record done!")


# if __name__ == "__main__":
#     main()


"""
ann_labels = [
    AnnotationLabel(0, " ", "NOTANN", "Not an actual annotation"),
    AnnotationLabel(1, "N", "NORMAL", "Normal beat"),
    AnnotationLabel(2, "L", "LBBB", "Left bundle branch block beat"),
    AnnotationLabel(3, "R", "RBBB", "Right bundle branch block beat"),
    AnnotationLabel(4, "a", "ABERR", "Aberrated atrial premature beat"),
    AnnotationLabel(5, "V", "PVC", "Premature ventricular contraction"),
    AnnotationLabel(6, "F", "FUSION", "Fusion of ventricular and normal beat"),
    AnnotationLabel(7, "J", "NPC", "Nodal (junctional) premature beat"),
    AnnotationLabel(8, "A", "APC", "Atrial premature contraction"),
    AnnotationLabel(9, "S", "SVPB", "Premature or ectopic supraventricular beat"),
    AnnotationLabel(10, "E", "VESC", "Ventricular escape beat"),
    AnnotationLabel(11, "j", "NESC", "Nodal (junctional) escape beat"),
    AnnotationLabel(12, "/", "PACE", "Paced beat"),
    AnnotationLabel(13, "Q", "UNKNOWN", "Unclassifiable beat"),
    AnnotationLabel(14, "~", "NOISE", "Signal quality change"),
    AnnotationLabel(15, None, None, None),
    AnnotationLabel(16, "|", "ARFCT", "Isolated QRS-like artifact"),
    AnnotationLabel(17, None, None, None),
    AnnotationLabel(18, "s", "STCH", "ST change"),
    AnnotationLabel(19, "T", "TCH", "T-wave change"),
    AnnotationLabel(20, "*", "SYSTOLE", "Systole"),
    AnnotationLabel(21, "D", "DIASTOLE", "Diastole"),
    AnnotationLabel(22, '"', "NOTE", "Comment annotation"),
    AnnotationLabel(23, "=", "MEASURE", "Measurement annotation"),
    AnnotationLabel(24, "p", "PWAVE", "P-wave peak"),
    AnnotationLabel(25, "B", "BBB", "Left or right bundle branch block"),
    AnnotationLabel(26, "^", "PACESP", "Non-conducted pacer spike"),
    AnnotationLabel(27, "t", "TWAVE", "T-wave peak"),
    AnnotationLabel(28, "+", "RHYTHM", "Rhythm change"),
    AnnotationLabel(29, "u", "UWAVE", "U-wave peak"),
    AnnotationLabel(30, "?", "LEARN", "Learning"),
    AnnotationLabel(31, "!", "FLWAV", "Ventricular flutter wave"),
    AnnotationLabel(32, "[", "VFON", "Start of ventricular flutter/fibrillation"),
    AnnotationLabel(33, "]", "VFOFF", "End of ventricular flutter/fibrillation"),
    AnnotationLabel(34, "e", "AESC", "Atrial escape beat"),
    AnnotationLabel(35, "n", "SVESC", "Supraventricular escape beat"),
    AnnotationLabel(36, "@", "LINK", "Link to external data (aux_note contains URL)"),
    AnnotationLabel(37, "x", "NAPC", "Non-conducted P-wave (blocked APB)"),
    AnnotationLabel(38, "f", "PFUS", "Fusion of paced and normal beat"),
    AnnotationLabel(39, "(", "WFON", "Waveform onset"),
    AnnotationLabel(40, ")", "WFOFF", "Waveform end"),
    AnnotationLabel(41, "r", "RONT", "R-on-T premature ventricular contraction"),
    AnnotationLabel(42, None, None, None),
    AnnotationLabel(43, None, None, None),
    AnnotationLabel(44, None, None, None),
    AnnotationLabel(45, None, None, None),
    AnnotationLabel(46, None, None, None),
    AnnotationLabel(47, None, None, None),
    AnnotationLabel(48, None, None, None),
    AnnotationLabel(49, None, None, None),
]
"""