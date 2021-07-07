from .utils import report_failed, report_exception, report_running, validate_node

def analyze_workchain(
    node,
    report_actions={},
    # **kwargs
    on_killed=None,
    on_running=report_running,
    on_finished_ok=None,
    on_failed=None,
    on_excepted=None,
    tab='',
    ):
    wc = validate_node(node)

    if wc.is_killed:
        print(tab + 'Killed!!!')
        if not on_killed is None:
            return on_killed(wc)
        return

    if wc.is_finished_ok:
        print(tab + 'finished_ok')
        if not on_finished_ok is None:
            return on_finished_ok(wc)
        return

    if wc.is_failed:
        print(tab + 'Failed:')
        report_failed(wc, tab=tab+'  ', actions=report_actions)

        if not on_failed is None:
            return on_failed(wc)
        return

    if wc.is_excepted:
        print(tab + 'Excepted!!')
        report_exception(wc, tab=tab+'  ')

        if not on_excepted is None:
            return on_excepted(wc)
        return

    print(tab + 'still running')
    if not on_running is None:
        return on_running(wc)