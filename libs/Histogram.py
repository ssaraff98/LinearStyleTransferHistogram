######################### ADDED #########################
import torch

def fixedWidthHistogram(f, range, bins):
    scaled_f = torch.div(f - range[0], range[1] - range[0])

    i = torch.floor(scaled_f * float(bins))
    i = torch.clamp(i, 0, float(bins) - 1)
    i = i.type(torch.FloatTensor)

    # counts = torch.scatter_add(torch.zeros_like(i, dtype=torch.float32), bins, i, torch.ones_like(i, dtype=torch.float32))
    counts = torch.ones_like(i, dtype=torch.float32)
    return counts, i

def sortSearch(x, search_values, bins):
    i = torch.zeros_like(search_values, dtype=dtype.float32)

    while bins > 1:
        bins /= 2

        iL = i
        iR = i + bins

        predicted = torch.less(search_values, f[iR])
        i = torch.where(predicted, iL, iR)

    predicted = torch.less(search_values, f[0])
    i = torch.where(predicted, i, i + 1)
    return i

def linearInterpolation(newF, f, y, bins):
    new_i = sortSearch(f, newF, bins)

    min = new_i - 1
    max = new_i

    max = torch.clamp(max, 0, bins - 1)
    min = torch.clamp(min, 0, bins - 1)

    min_f = f[min]
    max_f = f[max]
    min_y = y[min]
    max_y = y[max]

    slope = (max_y - min_y) / (max_f - min_f)
    linear_y = slope * (newF - min_f) + min_y
    closest_y = min_y

    newY = (torch.not_equal((max_f - min_f), 0.0), linear_y, closest_y)
    return newY

def matchHistogram(sF, tF, bins):
    s_range = [torch.min(sF), torch.max(sF)]
    t_range = [torch.min(tF), torch.max(tF)]

    t_values = torch.linspace(t_range[0].item(), t_range[1].item(), bins)

    s_counts, i = fixedWidthHistogram(sF, s_range, bins)
    t_counts, _ = fixedWidthHistogram(tF, t_range, bins)

    s_counts_dF = torch.cumsum(s_counts, bins)
    s_counts_dF /= s_counts_dF[-1]

    t_counts_dF = torch.cumsum(t_counts.float())
    t_counts_dF /= t_counts_dF[-1]

    interpolated_t = linearInterpolation(s_counts_dF, t_counts_dF, t_values, bins)
    interpolated_t = torch.max(interpolated_t, 0.0)

    gathered_t = interpolated_t[i]
    return gathered_t
######################### ADDED #########################
