import torch
def flow_matching_loss_pairs(
    model,
    z0,
    z1,
    cond
):

    t = torch.rand(
        z0.shape[0],
        1,
        device=z0.device
    )

    zt = (1 - t) * z0 + t * z1

    target_v = z1 - z0

    pred_v = model(
        zt,
        t,
        cond
    )

    return ((pred_v - target_v) ** 2).mean()

def flow_matching_loss_gaussian(

    model,
    z0,
    z1,
    cond,
    sigma=0.2
):

    t = torch.rand(

        z0.shape[0],
        1,

        device=z0.device
    )


    mean = (1-t)*z0 + t*z1


    std = sigma * torch.sqrt(

        t*(1-t)

    )


    noise = torch.randn_like(mean)


    zt = mean + std*noise


    target_v = z1 - z0


    pred_v = model(

        zt,
        t,
        cond
    )


    return ((pred_v - target_v)**2).mean()