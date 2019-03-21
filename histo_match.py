import numpy as np

def hist_match(source_in, template_in, channel):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source_in: list of 2D np.ndarrays
            Image to transform; the histogram is computed over the flattened
            array

        template: np.ndarray
            Template image; can have different dimensions to source

        channel: int
            which channel of the template will be used for adjustment

    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    out = []
    for i in range(len(source_in)):
        oldshape = source_in[i].shape
        source = source_in[i].ravel()
        template = template_in[channel].ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        out.append(interp_t_values[bin_idx].reshape(oldshape))

    return out