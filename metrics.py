import numpy as np
'''
including all metrics used in the paper
1. mean error
2. success rate
'''

'''
compute sucess rate in % for given threshold
compared with original prediction
'''


def success_rate(mean_serror, mean_terror, hyper, logger):
    logger.info('--------------success rate in % on average----------------')
    s_threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    t_threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.73, 0.748, 0.8, 0.9]

    for s in s_threshold:
        srate = np.sum(mean_serror > s) / len(mean_serror) * 100
        logger.info('{0}% of frames have steer error > {1} '.format(np.round(srate, 4), s * 70))

    for t in t_threshold:
        trate = np.sum(mean_terror > t) / len(mean_terror) * 100
        logger.info('{0}% of frames have throttle error > {1} '.format(np.round(trate, 4), t * 46))

'''
compute mean error and median error metrics
'''
def compute_metric(hyper, serror_framelist, terror_framelist, berror_framelist, logger):
    logger.info('----ME {0} for {1}, iters={2} ----'.format(hyper['title'], hyper['method'], hyper['iters']))
    ME_s = np.mean(serror_framelist)
    ME_t = np.mean(terror_framelist)
    logger.info('s={0}, t={1}'.format(ME_s * 70, ME_t * 46))

    logger.info('-----median error {0} for {1}-----'.format(hyper['title'], hyper['method']))
    median_s = np.median(serror_framelist)
    median_t = np.median(terror_framelist)
    logger.info('s={0}, t={1}'.format(median_s * 70, median_t * 46))