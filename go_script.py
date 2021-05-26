import pandas as pd
import time
import datetime
import logging

import constants
from loss_rates import LossRateMatrices, SimplifiedMethodologyLossRate, Biz2CreditLossRate, IOULossRate
from recovery_rate import get_recovery_rate
from data_mapping_final import run_mapping
from overrides import generate_haircut_results

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


"""
This is the main entry point into the code to run the monthly valuations. See documentation.rst for instructions.
"""


################################################################################################
##############  SETTINGS   #####################################################################
################################################################################################
'''
PROSPER = 'Prosper'
LC_PRIME_36 = 'lc_prime_36'
LC_NEAR_PRIME_36 = 'lc_near_prime_36'
LC_NEAR_PRIME_60 = 'lc_near_prime_60'
BIZ2CREDIT = 'biz2credit'
IOU_LOANS = 'iou'
'''

# TODO create the following two folders and load data into the Biz2Credit Data
# 'C:\202006_AllData\Biz2Credit'
# 'C:\202006_AllData\Results'

DATA_DATE = 202006  # Date of when the data is from
RUN_TYPE = constants.BIZ2CREDIT
RUN_DATE = 201911  # Date to run of the valuation

# Mapping is for LC, Biz2Credit & IOU Loans
generate_mapping = False

# These 3 are for LC & Prosper
generate_statuses = True
generate_transition_matrices = False
generate_principal_outstanding = False

# These 2 are for everyone
generate_universal_recovery_rate = True
generate_loss_rates_at_maturity = True


# Haircuts are for everyone
generate_haircuts = True

# Only when running backtesting
run_backtesting = False
BACKTESTING_DATES_STRING = [datetime.date(x,y,1).strftime('%Y%m') for x in range(2013,2020) for y in range(1,13)]
BACKTESTING_DATES = [int(x) for x in BACKTESTING_DATES_STRING]
run_with_prosper = False

################################################################################################


# Get the file names
logging.info('Setting up file names')
BASE_PATH = 'C:/%s_AllData/' % str(DATA_DATE)

full_input_data_path = None
file_end = None
if RUN_TYPE == constants.BIZ2CREDIT:
    file_end = 'biz2credit'
    full_input_data_path = BASE_PATH + '/Results/Biz2Credit_Full_Data.csv'
elif RUN_TYPE == constants.IOU_LOANS:
    file_end = 'iou'
    full_input_data_path = BASE_PATH + '/Results/iou.csv'

output_files_dir = BASE_PATH + 'Results/'
statuses_file = 'all_statuses_%s.pkl' % file_end
transition_matrices_file = 'transition_matrices_%s.pkl' % file_end
loss_rates_at_maturity_file = 'loss_rates_at_maturity_%s.pkl' % file_end
principal_outstanding_file = 'principal_outstanding_%s.pkl' % file_end
recovery_rates_file = 'recovery_rates_%s.pkl' % file_end
haircuts_file = 'haircuts_%s.pkl' % file_end
haircuts_file_csv = 'haircuts_%s.csv' % file_end

if RUN_TYPE in constants.SIMPLIFIED_METHODOLOGY_METHOD:
    generate_transition_matrices = False
    generate_principal_outstanding = False

if RUN_TYPE in [constants.IOU_LOANS, constants.BIZ2CREDIT]:
    generate_statuses = False
    dates_to_run = [RUN_DATE] if isinstance(RUN_DATE, int) else list(RUN_DATE)
else:
    RUN_DATE = DATA_DATE
    dates_to_run = False

if run_backtesting:
    backtesting_path = BASE_PATH + '/Results/%s_Backtesting_Results.csv' %RUN_TYPE
    dates_to_run = BACKTESTING_DATES

start = time.time()

if generate_mapping:
    # Automated running the mapping
    logging.info('Running mapping for %s' % RUN_TYPE)
    run_mapping(RUN_TYPE, DATA_DATE, BASE_PATH, BASE_PATH + 'Results/')


if generate_loss_rates_at_maturity:
    # Generate loss rates at maturity for each measurement date.
    # First generate data set that can be used to generate loss rates at maturity.
    # Filters the full data set to get just the required columns. This part should probably be added to loss_rates.py
    cols = [constants.LOANID,
            constants.LOANAMOUNT,
            constants.GRADE,
            constants.TERM,
            constants.CHARGEOFF,
            constants.AGE,
            constants.MONTH,
            constants.CUMUL_PRIN,
            constants.DEFAULTMONTH,
            constants.COMPLETEDMONTH,
            constants.CHARGEOFFMONTH,
            constants.ORIGMID,
            constants.DPD]

    logging.info('Generating loss rates at maturity....')
    if RUN_TYPE in [constants.LC_NEAR_PRIME_60]:
        df = pd.read_csv(full_input_data_path) # the seperate pickle reads are due to memory considerations
        logging.info('Running simplified methodology')
        loss_rates = SimplifiedMethodologyLossRate(df, RUN_TYPE)
        df_lr = loss_rates.generate_backtesting_loss_rates()
    elif RUN_TYPE in [constants.BIZ2CREDIT]:
        df = pd.read_csv(full_input_data_path) #Hack
        logging.info('Running Biz2Credit Methodology')
        loss_rates = Biz2CreditLossRate(df, RUN_TYPE, run_with_prosper)
        df_lr = loss_rates.generate_backtesting_loss_rates(dates_to_run)
        df_lr = pd.Series(df_lr)
    elif RUN_TYPE in [constants.IOU_LOANS]:
        df = pd.read_csv(full_input_data_path)
        logging.info('Running IOU Methodology')
        loss_rates = IOULossRate(df, RUN_TYPE, run_with_prosper)
        df_lr = loss_rates.generate_backtesting_loss_rates(dates_to_run)
        df_lr = pd.Series(df_lr)
    else:
        df = pd.read_csv(full_input_data_path, usecols=cols)
        loss_rates = LossRateMatrices(df, RUN_TYPE)
        df_lr = loss_rates.generate_all_matrices()


    df_lr.to_pickle(output_files_dir + loss_rates_at_maturity_file)
    logging.info('Complete')


if generate_universal_recovery_rate:
    logging.info('Generating universal recovery rate...')
    rr = get_recovery_rate(full_input_data_path, RUN_TYPE, dates_to_run)
    pd.Series(rr).to_pickle(output_files_dir + recovery_rates_file)
    logging.info('Complete')

logging.info('Time taken:')
logging.info((time.time() - start) / 60.)

if generate_haircuts:
    logging.info('Building final output results...')

    df = generate_haircut_results(RUN_DATE, RUN_TYPE, output_files_dir, transition_matrices_file,
                                  principal_outstanding_file, loss_rates_at_maturity_file, recovery_rates_file,
                                  run_with_prosper)

    df.to_csv(output_files_dir + haircuts_file_csv)


if run_backtesting:
    df = pd.DataFrame()
    for date in BACKTESTING_DATES:
        new_df = generate_haircut_results(date, RUN_TYPE, output_files_dir, transition_matrices_file,
                                  principal_outstanding_file, loss_rates_at_maturity_file, recovery_rates_file, run_with_prosper)
        new_df['BacktestingDate'] = date
        df = df.append(new_df)
    df.to_csv(backtesting_path)