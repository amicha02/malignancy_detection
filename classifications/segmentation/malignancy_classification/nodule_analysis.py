from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
logging.getLogger("p2ch13.dsets").setLevel(logging.WARNING)
logging.getLogger("p2ch14.dsets").setLevel(logging.WARNING)



class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]



if __name__ == '__main__':
    NoduleAnalysisApp().main()