import torch
import numpy as np
from scripts.pairing import encode_condition
def sample_flow(
    model,
    z0,
    cond,
    n_steps=300
):

    device = next(model.parameters()).device

    z = z0.to(device)

    cond = cond.to(device)

    dt = 1.0/n_steps

    for i in range(n_steps):

        t = torch.ones(

            len(z),

            1,

            device=device

        ) * (i/n_steps)

        v = model(
            z,
            t,
            cond
        )

        z = z + dt*v

    return z
def get_control_latent(pert, z_NC, z_NCNC):
    
    n_genes = len(pert.split("+"))

    if n_genes == 1:
        return z_NC

    return z_NCNC

def sample_all_perts(

    model,

    pert_list,

    z_NC,

    z_NCNC,

    n_steps=300,

):

    device = next(model.parameters()).device


    all_latent = []

    pert_labels = []


    for pert in pert_list:

        ##################################
        # choose correct control
        ##################################

        z_source_np = get_control_latent(

            pert,

            z_NC,

            z_NCNC

        )


        z_source = torch.tensor(

            z_source_np,

            dtype=torch.float32,

            device=device

        )


        ##################################
        # condition vector
        ##################################

        cond_vec = encode_condition(pert)

        cond_vec = torch.tensor(

            cond_vec,

            dtype=torch.float32,

            device=device

        ).unsqueeze(0)


        cond_batch = cond_vec.repeat(

            len(z_source),

            1

        )


        ##################################
        # flow sampling
        ##################################

        z_pred = sample_flow(

            model,

            z_source,

            cond_batch,

            n_steps=n_steps
        )


        z_pred = z_pred.detach().cpu().numpy()


        ##################################
        # store results
        ##################################

        all_latent.append(z_pred)

        pert_labels.extend(

            [pert] * len(z_pred)

        )


        print(

            f"{pert:20s}",

            "source:",

            "NC" if len(pert.split("+"))==1 else "NC+NC",

            z_pred.shape

        )


    ##################################
    # concatenate
    ##################################

    latent_all = np.vstack(all_latent)


    return latent_all, pert_labels