import datetime
from util.util import importstr
from util.logconf import logging
import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'
log = logging.getLogger('nb')

def run(app, *argv):
    argv = list(argv)

    log.info("Running: {}({!r}).main()".format(app, argv)) 
    app_cls = importstr(*app.rsplit('.', 1)) 
    app_cls(argv).main()
    log.info("Finished: {}.{!r}).main()".format(app, argv))



if __name__ == "__main__":
    run('nodule_analysis.NoduleAnalysisApp','series_uid=1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260')

 