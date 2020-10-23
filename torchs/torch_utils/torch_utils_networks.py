def update_one_from_the_other(source, target, tau):
    for params_source, params_target in zip(source.parameters(), target.parameters()):
        params_target.data.copy_(tau * params_source.data + (1 - tau) * params_target)

