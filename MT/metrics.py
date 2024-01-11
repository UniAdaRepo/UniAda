import numpy as np


def success_rate(mean_serror, mean_terror, logger):
    logger.info('--------------success rate in % on average----------------')
    mean_serror = mean_serror * 70
    mean_terror = mean_terror * 1.609344
    s_threshold = [3.5, 14, 21, 28]
    t_threshold = [4.6, 13.8, 23.0, 32.2]

    for s in s_threshold:
        srate = np.sum(mean_serror > s) / len(mean_serror) * 100
        logger.info('{0}% of frames have steer error > {1} '.format(np.round(srate, 4), s))


    for t in t_threshold:
        trate = np.sum(mean_terror > t) / len(mean_terror) * 100
        logger.info('{0}% of frames have throttle error > {1} '.format(np.round(trate, 4), t))

'''
compute mean error and median error metrics
'''
def compute_metric(hyper, serror_framelist, terror_framelist, logger):
    logger.info('----ME {0} for {1}, iters={2} ----'.format(hyper['title'], hyper['method'], hyper['iters']))
    logger.info('steer error framelist = {0}'.format(serror_framelist))
    ME_s = np.mean(serror_framelist)
    ME_t = np.mean(terror_framelist)
    logger.info('s={0}, t={1}'.format(ME_s * 70, ME_t * 1.609344))

    logger.info('-----median error {0} for {1}-----'.format(hyper['title'], hyper['method']))
    median_s = np.median(serror_framelist)
    median_t = np.median(terror_framelist)
    logger.info('s={0}, t={1}'.format(median_s * 70, median_t * 1.609344))