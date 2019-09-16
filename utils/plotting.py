
import matplotlib.pyplot as plt
import numpy as np

def plot2DdataImage(data, title,figure =None,  ratio = 0.5,
                    subplot_position =111,show=False, colorbar =True ):
    if figure is None:
        figure = plt.figure()
    ax = figure.add_subplot(subplot_position)
    im = ax.imshow(data, aspect=float(data.shape[1]) /
                                data.shape[0] * ratio,
                   interpolation='nearest' ,  cmap='magma')
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im)
    if show:
        plt.show()

def plot_marginals(m, v_unitless, s, k, var, std_dev, title):
    fig, axes = plt.subplots(3, 2)
    plt.suptitle(title)
    axes[0, 0].plot(m)
    axes[0, 0].set_title("mean")
    axes[1, 0].plot(var)
    axes[1, 0].set_title("variance")
    axes[2, 0].plot(v_unitless)
    axes[2, 0].set_title("unitless variance")
    axes[0, 1].plot(s)
    axes[0, 1].set_title("skew")
    axes[1, 1].plot(k)
    axes[1, 1].set_title("kurtosis")
    axes[2, 1].plot(std_dev)
    axes[2, 1].set_title("standard dev")

    features = np.vstack((m, v_unitless, s, k))
    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    plt.imshow(features)
    plt.title("ERB filterbank marginal statistics")

    plt.subplot(212)
    norm = np.linalg.norm(features, axis=1, ord=np.inf)
    norm_features = features / norm[:, None]
    plt.imshow(norm_features)
    plt.title("Normalized marginal statistics")

def plot_marginal(title, mean, norm_var,skew,kurtosis, size = (6,3)):
    f, ax = plt.subplots(2, 2, sharex='col', figsize=size)
    f.suptitle(title)
    ax[0,0].plot(mean, label="mean")
    ax[0,0].set_title("Mean")

    ax[0,1].plot(norm_var, label="Unitless variance $\sigma^2/\mu^2$")
    ax[0,1].set_title("Unitless variance $\sigma^2/\mu^2$")

    ax[1,0].plot(skew, label="skew")
    ax[1,0].set_title("Skewness")
    ax[1,0].set_xlabel("Filterbank channel")

    ax[1,1].plot(kurtosis, label="kurtosis")
    ax[1,1].set_title("Kurtosis")
    ax[1,1].set_xlabel("Filterbank channel")
    plt.tight_layout()
    return f, ax

def plot_correlation(corr, title, close =True,  size = (4,4)):
    f, ax =plt.subplots(1, 1, figsize=size)
    ax.set_title(title)
    im = ax.imshow(corr)
    f.colorbar(im, ax=ax)
    if close:
        plt.close()
    return f, ax

def plot_filterbank_channels_second_stage(am_env, minChannel, maxChannel, close =True, size = (2, 1)):
    axes = []
    for band_index in range(minChannel, maxChannel):
        f, ax = plt.subplots(1, 1, figsize=size)

        ax.set_title("AM envelopes for band # {}".format(band_index))
        im = ax.imshow(am_env[band_index], aspect=float(am_env[band_index].shape[1]) / am_env[band_index].shape[0] * 0.5,
                   interpolation='nearest')
        f.colorbar(im, ax=ax)
        axes.append(ax)
        if close:
            plt.close()
    return axes

def plot_modulation_power(mp, axes =None, close =True, size = (3,4)):
    f, ax =plt.subplots(1, 1, figsize=size)
    ax.set_title("Modulation power")
    im = ax.imshow(mp)
    f.colorbar(im, ax=ax)
    ax.set_xlabel("Modulation band")
    ax.set_ylabel("Frequency  band")
    if axes !=None:
        ax.set_xticks(np.arange(len(axes[1])))
        ax.set_yticks(np.arange(len(axes[0])))
        ax.set_xticklabels( axes[1], rotation = 45)
        ax.set_yticklabels( axes[0])
        print("axes")
    if close:
        plt.close()
    return f,ax

def plot_individual_channel(channels, minChannel, maxChannel, close =True, size = (2, 1)):
    axes = []

    for channel_index in range(minChannel, maxChannel):
        f, ax = plt.subplots(1, 1, figsize=size)
        ax.plot(channels[channel_index])
        ax.set_title("Channel  # {}".format(channel_index))
        axes.append(ax)
        if close:
            plt.close()

    return axes

def plot_windows(windows, wins, leng, sr, close =True, size = (6,3), show=False):
    f, ax = plt.subplots(1,1, figsize=size)
    centers= np.zeros((leng))
    cm = plt.get_cmap("tab20")
    for index, (g, window) in enumerate(zip(windows[:], wins[:])) :
        if index < len(windows)/2:
            colorIndex =index
        else: colorIndex=  len(windows)//2 - (index - len(windows)//2)

        centers[window[len(window)//2]] =1
        filter = np.fft.fftshift(g)
        t_window = np.zeros(leng)
        t_window[window] = filter
        ax.plot(np.fft.fftshift(t_window), color=cm.colors[colorIndex%cm.N])
    centers = np.fft.fftshift(centers)
    centers= np.argwhere(centers==1).reshape(-1)

    frequencies = ((np.array(centers)- leng/2) *sr/leng).astype(np.int32)
    # ax.vlines(leng/2,ymin=0, ymax =1., colors='r')
    ax.set_xticks( centers)
    ax.set_xticklabels(frequencies, rotation =45)
    plt.title("Filters placement on frequency axis")
    if close:
        plt.close()
    if show:
        plt.show()
    return f, ax


def plotCenterFrequencies(centerFrequencies, cfPositions):
    plt.figure()
    plt.vlines(centerFrequencies, ymin=0, ymax=1)
    plt.title("CenterFrequencies")

    plt.figure()
    plt.vlines(cfPositions, ymin=0, ymax=1)
    plt.title("Center positions")

    plt.show()