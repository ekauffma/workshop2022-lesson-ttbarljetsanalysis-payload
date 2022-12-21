#import asyncio
import time
#import logging

import awkward as ak

from coffea import processor

from utils import *

from coffea.nanoevents import NanoAODSchema

#### GLOBAL CONFIGURATION

# input files per process per sample
# -1  means running over all datasets
N_FILES_MAX_PER_SAMPLE = -1

PIPELINE = "coffea"

USE_DASK = True

# analysis facility: set to "coffea_casa" for coffea-casa environments, "EAF" for FNAL, "local" for local setups
AF = "coffea_casa"


processor_base = processor.ProcessorABC


# functions creating systematic variations
#-----------------------------
def flat_variation(ones):
#-----------------------------
    # 0.1% weight variations
    return (1.0 + np.array([0.001, -0.001], dtype=np.float32)) * ones[:, None]

#-----------------------------------------
def btag_weight_variation(i_jet, jet_pt):
#-----------------------------------------
    # weight variation depending on i-th jet pT (10% as default value, multiplied by i-th jet pT / 50 GeV)
    return 1 + np.array([0.1, -0.1]) * (ak.singletons(jet_pt[:, i_jet]) / 50).to_numpy()

#---------------------------
def jet_pt_resolution(pt):
#---------------------------
    # normal distribution with 5% variations, shape matches jets
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.05)
    return ak.unflatten(resolution_variation, counts)

#------------------------------------
class TtbarAnalysis(processor_base):
#------------------------------------
    #--------------------
    def __init__(self):
    #--------------------
        num_bins = 25
        bin_low = 50
        bin_high = 550
        name = "observable"
        label = "observable [GeV]"
        #https://hist.readthedocs.io/en/latest/user-guide/quickstart.html
        #StrCat = StrCategory
        #https://hist.readthedocs.io/en/latest/banner_slides.html?highlight=StrCategory#many-axis-types
        self.hist = (
            hist.Hist.new.Reg(num_bins, bin_low, bin_high, name=name, label=label)
            .StrCat(["4j1b", "4j2b"], name="region", label="Region")
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="variation", label="Systematic variation", growth=True)
            .Weight()
        )
    #-------------------------
    def process(self, events):
    #-------------------------
        histogram = self.hist.copy()

        process = events.metadata["process"]  # "ttbar" etc.
        variation = events.metadata["variation"]  # "nominal", "scaledown", etc.

        #print(f'Currently doing variation {variation} for {process}')

        # normalization for MC
        x_sec = events.metadata["xsec"]
        nevts_total = events.metadata["nevts"]
        # This lumi was obtained with
        # brilcalc lumi -c web -i Cert_13TeV_16Dec2015ReReco_Collisions15_25ns_JSON_v2.txt -u /pb --normtag normtag_PHYSICS_2015.json  --begin 256630 --end 260627 > lumi2015D.txt
        # lumi in units of /pb
        lumi = 2256.38
        if process != "data":
            xsec_weight = x_sec * lumi / nevts_total
        else:
            xsec_weight = 1

        #### systematics
        # example of a simple flat weight variation, using the coffea nanoevents systematics feature
        # https://github.com/CoffeaTeam/coffea/blob/20a7e749eea3b8de4880088d2f0e43f6ef9d7993/coffea/nanoevents/methods/base.py#L84
        # Help on method add_systematic in module coffea.nanoevents.methods.base:
        # add_systematic(name: str, kind: str, what: Union[str, List[str], Tuple[str]], varying_function: Callable)
        # method of coffea.nanoevents.methods.base.NanoEventsArray instance
        if process == "wjets":
            events.add_systematic("scale_var", "UpDownSystematic", "weight", flat_variation)

        # example on how to get jet energy scale / resolution systematics
        # need to adjust schema to instead use coffea add_systematic feature, especially for ServiceX
        # cannot attach pT variations to events.jet, so attach to events directly
        # and subsequently scale pT by these scale factors
        events["pt_nominal"] = 1.0
        events["pt_scale_up"] = 1.03
        # we have already these corrections in our data for this workshop, so we might as well use them
        # to assign a variation per jet and not per event. However, to avoid messing too much with this code, 
        # try a silly thing just for fun: take the average of jet variations per event (fill out the None values with a default 1.03)
        # events['pt_scale_up'] = ak.fill_none(ak.mean(events.Jet.corrptUp/events.Jet.corrpt,axis=1),1.03)
        events["pt_res_up"] = jet_pt_resolution(events.Jet.pt)

        pt_variations = ["pt_nominal", "pt_scale_up", "pt_res_up"] if variation == "nominal" else ["pt_nominal"]
        for pt_var in pt_variations:

            ### event selection
            # based on https://link.springer.com/article/10.1007/JHEP09(2017)051

            #object filters
            selected_electrons = events.Electron[(events.Electron.pt > 30) & (abs(events.Electron.eta)<2.1) & (events.Electron.cutBased == 4) & (events.Electron.sip3d < 4)]
            selected_muons = events.Muon[(events.Muon.pt > 30) & (abs(events.Muon.eta)<2.1) & (events.Muon.tightId == True) & (events.Muon.sip3d < 4) & (events.Muon.pfRelIso04_all < 0.15)]
            jet_filter = (events.Jet.pt * events[pt_var] > 30) & (abs(events.Jet.eta)<2.4)
            selected_jets = events.Jet[jet_filter]

            # single lepton requirement
            event_filters = ((ak.count(selected_electrons.pt, axis=1) + ak.count(selected_muons.pt, axis=1)) == 1)
            # at least four jets
            pt_var_modifier = events[pt_var] if "res" not in pt_var else events[pt_var][jet_filter]
            event_filters = event_filters & (ak.count(selected_jets.pt * pt_var_modifier, axis=1) >= 4)
            # at least one b-tagged jet ("tag" means score above threshold)
            B_TAG_THRESHOLD = 0.8
            event_filters = event_filters & (ak.sum(selected_jets.btagCSVV2 >= B_TAG_THRESHOLD, axis=1) >= 1)

            # apply event filters
            selected_events = events[event_filters]
            selected_electrons = selected_electrons[event_filters]
            selected_muons = selected_muons[event_filters]
            selected_jets = selected_jets[event_filters]

            for region in ["4j1b", "4j2b"]:
                # further filtering: 4j1b CR with single b-tag, 4j2b SR with two or more tags
                if region == "4j1b":
                    region_filter = ak.sum(selected_jets.btagCSVV2 >= B_TAG_THRESHOLD, axis=1) == 1
                    selected_jets_region = selected_jets[region_filter]
                    # use HT (scalar sum of jet pT) as observable
                    pt_var_modifier = events[event_filters][region_filter][pt_var] if "res" not in pt_var else events[pt_var][jet_filter][event_filters][region_filter]
                    observable = ak.sum(selected_jets_region.pt * pt_var_modifier, axis=-1)

                elif region == "4j2b":
                    region_filter = ak.sum(selected_jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2
                    selected_jets_region = selected_jets[region_filter]

                    # reconstruct hadronic top as bjj system with largest pT
                    # the jet energy scale / resolution effect is not propagated to this observable at the moment
                    trijet = ak.combinations(selected_jets_region, 3, fields=["j1", "j2", "j3"])  # trijet candidates
                    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
                    trijet["max_btag"] = np.maximum(trijet.j1.btagCSVV2, np.maximum(trijet.j2.btagCSVV2, trijet.j3.btagCSVV2))
                    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # require at least one-btag in trijet candidates
                    # pick trijet candidate with largest pT and calculate mass of system
                    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
                    observable = ak.flatten(trijet_mass)

                ### histogram filling
                if pt_var == "pt_nominal":
                    # nominal pT, but including 2-point systematics
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=variation, weight=xsec_weight
                        )


                    if variation == "nominal":
                        # also fill weight-based variations for all nominal samples
                        # this corresponds to the case for wjets included above as an example
                        for weight_name in events.systematics.fields:
                            for direction in ["up", "down"]:
                                # extract the weight variations and apply all event & region filters
                                weight_variation = events.systematics[weight_name][direction][f"weight_{weight_name}"][event_filters][region_filter]
                                # fill histograms
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )

                        # calculate additional systematics: b-tagging variations
                        for i_var, weight_name in enumerate([f"btag_var_{i}" for i in range(4)]):
                            for i_dir, direction in enumerate(["up", "down"]):
                                # create systematic variations that depend on object properties (here: jet pT)
                                if len(observable):
                                    weight_variation = btag_weight_variation(i_var, selected_jets_region.pt)[:, i_dir]
                                else:
                                    weight_variation = 1 # no events selected
                                histogram.fill(
                                    observable=observable, region=region, process=process, variation=f"{weight_name}_{direction}", weight=xsec_weight*weight_variation
                                )

                elif variation == "nominal":
                    # pT variations for nominal samples
                    histogram.fill(
                            observable=observable, region=region, process=process, variation=pt_var, weight=xsec_weight
                        )
        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist": histogram}

        return output
    # https://coffeateam.github.io/coffea/api/coffea.processor.ProcessorABC.html?highlight=postprocess#coffea.processor.ProcessorABC.postprocess
    def postprocess(self, accumulator):
        return accumulator




# "Fileset" construction and metadata via utils.py file
# Here, we gather all the required information about the files we want to process: paths to the files and asociated metadata
# making use of the utils.py code in this repository
fileset = construct_fileset(N_FILES_MAX_PER_SAMPLE, use_xcache=False, ntuples_file="ntuples_nanoaod_all.json")
print(f"processes in fileset: {list(fileset.keys())}")
t0 = time.time()



#Execute the data delivery pipeline
if PIPELINE == "coffea":
    
    if USE_DASK:
        executor = processor.DaskExecutor(client=get_client(AF))
        
    else:
        executor = processor.FuturesExecutor(workers = 4)

    schema = NanoAODSchema
    run = processor.Runner(executor=executor, schema=schema, savemetrics=True, metadata_cache={})
    all_histograms, metrics = run(fileset, "Events", processor_instance=TtbarAnalysis())
    all_histograms = all_histograms["hist"]


print(f"\nexecution took {time.time() - t0:.2f} seconds")

save_histograms(all_histograms, fileset, "histograms_all.root")