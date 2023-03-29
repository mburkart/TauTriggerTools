/*! Definition of a tuple with all event information that is required for the tau analysis.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#pragma once

#include "TauTriggerTools/Common/interface/SmartTree.h"
#include "TauTriggerTools/Common/interface/TauIdResults.h"
#include <Math/VectorUtil.h>

#define TAU_ID(name, pattern, has_raw, wp_list) VAR(uint16_t, name) VAR(Float_t, name##raw)

#define VAR2(type, name1, name2) VAR(type, name1) VAR(type, name2)
#define VAR3(type, name1, name2, name3) VAR2(type, name1, name2) VAR(type, name3)
#define VAR4(type, name1, name2, name3, name4) VAR3(type, name1, name2, name3) VAR(type, name4)

#define EVENT_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run number */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Int_t, npv) /* number of primary vertices */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    /* PF MET variables */ \
    VAR2(Float_t, met_pt, met_phi) /* pt and phi of the MET */ \
    /* Tag muon variables */ \
    VAR4(Float_t, muon_pt, muon_eta, muon_phi, muon_mass) /* 4-momentum of the muon */ \
    VAR(Int_t, muon_charge) /* muon charge */ \
    VAR(Float_t, muon_iso) /* muon pfRel isolation */ \
    VAR(Float_t, muon_mt) /* muon transverse mass */ \
    VAR(Int_t, muon_gen_match) /* matching of the muon with leptons on the generator level:
                                  Electron = 1, Muon = 2, TauElectron = 3, TauMuon = 4, Tau = 5, NoMatch = 6 */ \
    VAR(Int_t, muon_gen_charge) /* charge of the gen lepton that was matched with the muon */ \
    VAR4(Float_t, muon_gen_vis_pt, muon_gen_vis_eta, muon_gen_vis_phi, muon_gen_vis_mass) /* visible 4-momentum of the
                  gen lepton that was matched with the muon */ \
    /* Basic tau variables */ \
    VAR(UInt_t, tau_sel) /* how tau was selected */ \
    VAR4(Float_t, tau_pt, tau_eta, tau_phi, tau_mass) /* 4-momentum of the tau */ \
    VAR(Int_t, tau_charge) /* tau charge */ \
    VAR(Int_t, tau_gen_match) /* matching of the tau with leptons on the generator level:
                                 Electron = 1, Muon = 2, TauElectron = 3, TauMuon = 4, Tau = 5, NoMatch = 6 */ \
    VAR(Int_t, tau_gen_charge) /* charge of the gen lepton that was matched with the tau */ \
    VAR4(Float_t, tau_gen_vis_pt, tau_gen_vis_eta, tau_gen_vis_phi, tau_gen_vis_mass) /* visible 4-momentum of the
                  gen lepton that was matched with the tau */ \
    VAR4(Float_t, tau_gen_rad_pt, tau_gen_rad_eta, tau_gen_rad_phi, tau_gen_rad_energy) /* visible 4-momentum of the
                  initial state radiation emmited by the gen tau */ \
    VAR4(Int_t, tau_gen_n_charged_hadrons, tau_gen_n_neutral_hadrons, tau_gen_n_gammas, tau_gen_n_gammas_rad) /*
                number of charged and neutral hadrons, gammas and initial state radiation gammas produced by the tau
                decay at the generator level */ \
    /* Tau ID variables */ \
    VAR(Int_t, tau_decayMode) /* tau decay mode */ \
    VAR(Int_t, tau_oldDecayModeFinding) /* tau passed the old decay mode finding requirements */ \
    TAU_IDS() \
    /* Tau transverse impact paramters.
       See cmssw/RecoTauTag/RecoTau/plugins/PFTauTransverseImpactParameters.cc for details */ \
    VAR(Float_t, tau_dxy) /* tau signed transverse impact parameter wrt to the primary vertex */ \
    VAR(Float_t, tau_dxy_error) /* uncertainty of the transverse impact parameter measurement */ \
    VAR(Float_t, tau_ip3d) /* tau signed 3D impact parameter wrt to the primary vertex */ \
    VAR(Float_t, tau_ip3d_error) /* uncertainty of the 3D impact parameter measurement */ \
    VAR(Float_t, tau_dz) /* tau dz of the leadChargedHadrCand wrt to the primary vertex */ \
    VAR(Float_t, tau_dz_error) /* uncertainty of the tau dz measurement */ \
    /* mu-tau variables */ \
    VAR(Float_t, vis_mass) /* visible mu-tau mass */ \
    /* HLT results and objects */ \
    VAR(ULong64_t, hlt_accept) /* HLT accept bits */ \
    VAR(ULong64_t, hlt_acceptAndMatch) /* HLT accept & match bits */ \
    VAR(std::vector<UInt_t>, hltObj_types) /* types of the HLT object */ \
    VAR4(std::vector<Float_t>, hltObj_pt, hltObj_eta, hltObj_phi, hltObj_mass) /* 4-momentum of the HLT object */ \
    VAR(std::vector<ULong64_t>, hltObj_hasPathName) /* whatever the HLT object has a path name */ \
    VAR(std::vector<ULong64_t>, hltObj_isBestMatch) /* whatever the HLT object the best match for a path name */ \
    VAR(std::vector<UInt_t>, hltObj_miniAODIndex) /* index of the HLT object record in the MiniAOD */ \
    /* HLT filters */ \
    VAR(std::vector<UInt_t>, filter_hltObj) /* index of the HLT object */ \
    VAR(std::vector<UInt_t>, filter_hash) /* hash of the name of the filter */ \
    /* Matched L1 tau */ \
    VAR4(Float_t, l1Tau_pt, l1Tau_eta, l1Tau_phi, l1Tau_mass) /* 4-momentum of the L1 tau */ \
    VAR(Int_t, l1Tau_hwIso) /* integer "hardware" isolation value of the L1 tau */ \
    VAR(Int_t, l1Tau_hwQual) /* integer "hardware" quality value of the L1 tau */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_trigger, Event, EventTuple, EVENT_DATA, "events")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_trigger, EventTuple, EVENT_DATA)
#undef VAR
#undef VAR2
#undef VAR3
#undef VAR4
#undef EVENT_DATA
#undef TAU_ID

namespace tau_trigger {

template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }
template<>
constexpr float DefaultFillValue<float>() { return -999.; }
template<>
constexpr int DefaultFillValue<int>() { return -999; }

} // namespace tau_tuple
