#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:25:53 2020

@author: abduhu

one strand to couple strand braiding
*****

This module aims to transform a weave of weaves on single strand to
a weave of brading after splitting that strand.
"""
from copy import deepcopy


def describe(weave, s0, init_strand, final_strand):
    """
    writes a weave in terms of elementary braids in detail.

    Inputs:
        seq: List:
            list of weave even powers
        s0:
            the first sigma index of even braids
        init_strand:
            the first strand's index.
                If init_strand > 0, positive braid power
                    else negative braid power.

        final_strand:
            the last strand's index.If init_strand > 0, positive braid power
            else negative braid power.

    Returns:
        dict:
            'sigma': weave of braiding indices.
            'power': weave of braidng powers.
            'strand_rank': weave of woven strand positions.
    """

    if len(weave) <= 1:
        raise Exception("The weave should include more than 2 braids!")

    # dilute weave to elementary sigmas
    sigma_index = []
    sigma_power = []

    # core weaves
    current_sigma_index = s0
    for power in weave:
        power_sign = 1 if power > 0 else -1
        for ii in range(abs(power)):
            sigma_index.append(current_sigma_index)
            sigma_power.append(power_sign)

        # Flipping the sigma_index value
        if current_sigma_index == 1:
            current_sigma_index = 2
        else:
            current_sigma_index = 1

    # edge weaves
    # starting weaves
    power_sign = 1 if init_strand > 0 else -1
    if init_strand in [-1, 1]:
        sigma_index = [1] + sigma_index
        sigma_power = [power_sign] + sigma_power
    elif init_strand in [-3, 3]:
        sigma_index = [2] + sigma_index
        sigma_power = [power_sign] + sigma_power

    # ending weaves
    power_sign = 1 if final_strand > 0 else -1
    if final_strand in [-1, 1]:
        sigma_index = sigma_index + [1]
        sigma_power = sigma_power + [power_sign]
    elif final_strand in [-3, 3]:
        sigma_index = sigma_index + [2]
        sigma_power = sigma_power + [power_sign]

    weaves = {"sigma": sigma_index, "power": sigma_power}

    # determine woven strand position
    strand_index = []
    current_strand_index = abs(init_strand)
    for sigma in weaves["sigma"]:
        strand_index.append(current_strand_index)

        # find next strand index
        if current_strand_index in [1, 3]:
            current_strand_index = 2
        # else current_strand_index = 2
        elif sigma == 1:
            current_strand_index = 1
        else:
            current_strand_index = 3

    strand_index.append(abs(final_strand))

    return {"sigma": sigma_index,
            "power": sigma_power,
            "strand_rank": strand_index}


def uncouple_old(
    weave, s0, init_strand, final_strand, rank_increment,
    inv=False, to_str=False
):
    """
    splits the woven strand to two strands. It serves for transforming
    a weave on 3 anyons to braids on 4 anyons before combine it to other
    braids to construct two qubit gate.
    Inputs:
        rank_increment: int:
            to increase sigma indices.
    """
    seq = describe(weave, s0, init_strand, final_strand)

    if inv:
        seq = time_mirror(seq)

    new_seq = {"sigma": [], "power": []}

    for ii, sigma in enumerate(seq["sigma"]):

        if seq["strand_rank"][ii] == 3:

            new_seq["sigma"].append(2)
            new_seq["sigma"].append(3)
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])

        elif seq["strand_rank"][ii] == 1:

            new_seq["sigma"].append(2)
            new_seq["sigma"].append(1)
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])

        elif sigma == 1:

            new_seq["sigma"].append(1)
            new_seq["sigma"].append(2)
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])

        else:

            new_seq["sigma"].append(3)
            new_seq["sigma"].append(2)
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])

    for ii, sigma in enumerate(new_seq["sigma"]):
        new_seq["sigma"][ii] += rank_increment

    if to_str:
        # Write braid op in Latex language
        braid_op = {}
        for list in new_seq:
            braid_op[list] = new_seq[list][::-1]

        new_seq["str"] = deepcopy(new_seq["sigma"])
        for ii, sigma in enumerate(braid_op["sigma"]):
            p = new_seq["power"][ii]
            new_seq["str"][ii] = f"\\sigma_{sigma}^{p}"

        new_seq["str"] = "".join(new_seq["str"])
    return new_seq


def uncouple(
    weave,
    s0,
    init_strand,
    final_strand,
    rank_increment,
    inv=False,
    to_str=False,
    n_coupled_strands=2,
):
    """
    splits the woven strand to two strands. It serves for transforming
    a weave on 3 anyons to braids on 4 anyons before combine it to other
    braids to construct two qubit gate.
    Inputs:
        rank_increment: int:
            to increase sigma indices.
    """
    seq = describe(weave, s0, init_strand, final_strand)

    if inv:
        seq = time_mirror(seq)

    new_seq = {"sigma": [], "power": []}

    for ii, sigma in enumerate(seq["sigma"]):

        if seq["strand_rank"][ii] == 3:

            for jj in range(2, 2 + n_coupled_strands, 1):

                new_seq["sigma"].append(jj)
                new_seq["power"].append(seq["power"][ii])

        elif seq["strand_rank"][ii] == 1:

            for jj in range(n_coupled_strands, 0, -1):
                new_seq["sigma"].append(jj)
                new_seq["power"].append(seq["power"][ii])

        elif sigma == 1:

            for jj in range(1, n_coupled_strands + 1):
                new_seq["sigma"].append(jj)
                new_seq["power"].append(seq["power"][ii])

        else:

            for jj in range(n_coupled_strands + 1, 1, -1):
                new_seq["sigma"].append(jj)
                new_seq["power"].append(seq["power"][ii])

    for ii, sigma in enumerate(new_seq["sigma"]):
        new_seq["sigma"][ii] += rank_increment

    if to_str:
        # Write braid op in Latex language
        braid_op = {}
        for list in new_seq:
            braid_op[list] = new_seq[list][::-1]

        new_seq["str"] = deepcopy(new_seq["sigma"])
        for ii, sigma in enumerate(braid_op["sigma"]):
            p = new_seq["power"][ii]
            new_seq["str"][ii] = f"\\sigma_{sigma}^{p}"

        new_seq["str"] = "".join(new_seq["str"])
    return new_seq


def uncouple_all(
    weave, s0, init_strand, final_strand, rank_increment,
    inv=False, to_str=False
):
    """
    splits the woven strand to two strands. It serves for transforming
    a weave on 3 anyons to braids on 4 anyons before combine it to other
    braids to construct two qubit gate.
    Inputs:
        rank_increment: int:
            to increase sigma indices.
    """
    seq = describe(weave, s0, init_strand, final_strand)

    if inv:
        seq = time_mirror(seq)

    new_seq = {"sigma": [], "power": []}

    for ii, sigma in enumerate(seq["sigma"]):

        if seq["sigma"][ii] == 2:

            new_seq["sigma"].append(4)
            new_seq["sigma"].append(3)
            new_seq["sigma"].append(5)
            new_seq["sigma"].append(4)
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])

        elif seq["sigma"][ii] == 1:

            new_seq["sigma"].append(2)
            new_seq["sigma"].append(1)
            new_seq["sigma"].append(3)
            new_seq["sigma"].append(2)
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])
            new_seq["power"].append(seq["power"][ii])

        else:

            print("This function is bad designed!")

    for ii, sigma in enumerate(new_seq["sigma"]):
        new_seq["sigma"][ii] += rank_increment

    if to_str:
        # Write braid op in Latex language
        braid_op = {}
        for list in new_seq:
            braid_op[list] = new_seq[list][::-1]

        new_seq["str"] = deepcopy(new_seq["sigma"])
        for ii, sigma in enumerate(braid_op["sigma"]):
            p = new_seq["power"][ii]
            new_seq["str"][ii] = f"\\sigma_{sigma}^{p}"

        new_seq["str"] = "".join(new_seq["str"])
    return new_seq


def time_mirror(seq):
    """
    Literally inverses the braiding.
    """
    new_seq = {
        "sigma": deepcopy(seq["sigma"])[::-1],
        "power": deepcopy(seq["power"])[::-1],
    }
    if "strand_rank" in seq:
        new_seq["strand_rank"] = seq["strand_rank"][::-1]

    for ii, power in enumerate(new_seq["power"]):
        new_seq["power"][ii] = -power

    return new_seq


def seq_to_latex(weave, s0, init_strand, final_strand):
    """
    writes a weave sequence in latex form.
    """

    tex = ""
    s = deepcopy(s0)

    if abs(init_strand) == 1:

        if init_strand < 0:
            tex = tex + r"\sigma_1^{-1}"
        else:
            tex = tex + r"\sigma_1^{1}"

    elif abs(init_strand) == 3:

        if init_strand < 0:
            tex = tex + r"\sigma_2^{-1}"
        else:
            tex = tex + r"\sigma_2^{1}"

    for power in weave:

        tex = rf"\sigma_{s}" + "^{" + f"{power}" + "}" + tex

        if s == 1:
            s = 2
        else:
            s = 1

    if abs(final_strand) == 1:

        if final_strand < 0:
            tex = r"\sigma_1^{-1}" + tex
        else:
            tex = r"\sigma_1^{1}" + tex

    elif abs(final_strand) == 3:

        if final_strand < 0:
            tex = r"\sigma_2^{-1}" + tex
        else:
            tex = r"\sigma_2^{1}" + tex

    return tex
