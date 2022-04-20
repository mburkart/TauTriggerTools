/*! Apply tau trigger selection vetoes.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TauTriggerTools/Common/interface/AnalysisTypes.h"
#include "TauTriggerTools/Common/interface/CutTools.h"
#include "TauTriggerTools/Common/interface/PatHelpers.h"

namespace tau_trigger {

class SelectionFilter : public edm::EDFilter {
public:
    using SelectionHist = root_ext::SmartHistogram<cuts::ObjectSelector>;
    using Cutter = cuts::Cutter<>;

    SelectionFilter(const edm::ParameterSet& cfg) :
        enabled(cfg.getParameter<bool>("enabled")),
        btagThreshold(cfg.getParameter<double>("btagThreshold")),
        mtCut(cfg.getParameter<double>("mtCut")),
        metFilters(cfg.getParameter<std::vector<std::string>>("metFilters")),
        electrons_token(consumes<pat::ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        jets_token(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
        met_token(consumes<pat::METCollection>(cfg.getParameter<edm::InputTag>("met"))),
        metFiltersResults_token(consumes<edm::TriggerResults>(cfg.getParameter<edm::InputTag>("metFiltersResults"))),
        selection("pre_selection")
    {
        const edm::ParameterSet& customMetFilters = cfg.getParameterSet("customMetFilters");
        for(const auto& filterName : customMetFilters.getParameterNames()) {
            customMetFilters_token[filterName] =
                mayConsume<bool>(customMetFilters.getParameter<edm::InputTag>(filterName));
        }
        produces<pat::MuonRefVector>();
    }

private:
    virtual bool filter(edm::Event& event, const edm::EventSetup&) override
    {
        if(!enabled) {
            event.put(std::make_unique<pat::MuonRefVector>());
            return true;
        }
        bool result = true;
        try {
            Cutter cut(&selection);
            filter(event, cut);
        } catch(cuts::cut_failed&) {
            result = false;
        }
        selection.fill_selection();
        return result;
    }

    virtual void endJob() override
    {
        TFile& file = edm::Service<TFileService>()->file();
        selection.SetOutputDirectory(&file);
        selection.WriteRootObject();
    }

    void filter(edm::Event& event, Cutter& cut)
    {
        cut(true, "tag_path_fired");
        edm::Handle<pat::MuonCollection> muons;
        event.getByToken(muons_token, muons);

        // Find signal muon
        std::vector<pat::MuonRef> signalMuonCandidates;
        for(size_t n = 0; n < muons->size(); ++n) {
            const pat::Muon& muon = muons->at(n);
            if(muon.polarP4().pt() > 24 && std::abs(muon.polarP4().eta()) < 2.1 && muon.isMediumMuon())
                signalMuonCandidates.emplace_back(muons, n);
        }
        cut(!signalMuonCandidates.empty(), "signal_muon");
        static const auto muonComparitor = [](const pat::MuonRef& a, const pat::MuonRef& b) {
            const double iso_a = MuonIsolation(*a), iso_b = MuonIsolation(*b);
            if(iso_a != iso_b) return iso_a < iso_b;
            return a->polarP4().pt() > b->polarP4().pt();
        };
        std::sort(signalMuonCandidates.begin(), signalMuonCandidates.end(), muonComparitor);
        const pat::Muon& signalMuon = *signalMuonCandidates.at(0);

        // Apply third lepton veto
        bool has_other_muon = false;
        for(const pat::Muon& muon : *muons) {
            if(&muon != &signalMuon && muon.isLooseMuon() && muon.polarP4().pt() > 10
                    && std::abs(muon.polarP4().eta()) < 2.4 && MuonIsolation(muon) < 0.3) {
                has_other_muon = true;
                break;
            }
        }
        cut(!has_other_muon, "muon_veto");

        edm::Handle<pat::ElectronCollection> electrons;
        event.getByToken(electrons_token, electrons);
        bool has_ele = false;
        for(const pat::Electron& ele : *electrons) {
            if(ele.polarP4().pt() > 10 && std::abs(ele.polarP4().eta()) < 2.5
                    && ele.electronID("mvaEleID-Fall17-iso-V2-wpLoose") > 0.5) {
                has_ele = true;
                break;
            }
        }
        cut(!has_ele, "ele_veto");

        // Apply MT cut (if enabled)
        if(mtCut > 0) {
            edm::Handle<pat::METCollection> metCollection;
            event.getByToken(met_token, metCollection);
            const pat::MET& met = metCollection->at(0);
            const analysis::LorentzVectorM met_p4(met.pt(), 0, met.phi(), 0);
            cut(Calculate_MT(signalMuon.polarP4(), met_p4) < mtCut, "mt_cut");
        }

        // Apply b tag veto (if enabled)
        if(btagThreshold > 0) {
            edm::Handle<pat::JetCollection> jets;
            event.getByToken(jets_token, jets);
            bool has_bjet = false;
            for(const pat::Jet& jet : *jets) {
                const auto btag = jet.bDiscriminator("pfDeepFlavourJetTags:probb")
                                  + jet.bDiscriminator("pfDeepFlavourJetTags:probbb")
                                  + jet.bDiscriminator("pfDeepFlavourJetTags:problepb");
                if(jet.polarP4().pt() > 20 && std::abs(jet.polarP4().eta()) < 2.4 && btag > btagThreshold) {
                    has_bjet = true;
                    break;
                }
            }
            cut(!has_bjet, "btag_veto");
        }

        // Apply MET filters
        edm::Handle<edm::TriggerResults> metFiltersResults;
        event.getByToken(metFiltersResults_token, metFiltersResults);
        const edm::TriggerNames& metFilterNames = event.triggerNames(*metFiltersResults);
        const auto passFilter = [&](const std::string& filter) {
            auto iter = customMetFilters_token.find(filter);
            if(iter != customMetFilters_token.end()) {
                edm::Handle<bool> result;
                event.getByToken(iter->second, result);
                return *result;
            }
            const size_t index = metFilterNames.triggerIndex(filter);
            if(index == metFilterNames.size())
                throw cms::Exception("TauTriggerSelectionFilter") << "MET filter '" << filter << "' not found.";
            return metFiltersResults->accept(index);
        };
        bool pass_met_filters = true;
        for(const std::string& metFilter : metFilters) {
            if(!passFilter(metFilter)) {
                pass_met_filters = false;
                break;
            }
        }
        cut(pass_met_filters, "met_filters");

        // Put the signal muon into the event
        auto signalMuonOutput = std::make_unique<pat::MuonRefVector>();
        signalMuonOutput->push_back(signalMuonCandidates.at(0));
        event.put(std::move(signalMuonOutput));
    }

private:
    const bool enabled;
    const double btagThreshold, mtCut;
    const std::vector<std::string> metFilters;

    edm::EDGetTokenT<pat::ElectronCollection> electrons_token;
    edm::EDGetTokenT<pat::MuonCollection> muons_token;
    edm::EDGetTokenT<pat::JetCollection> jets_token;
    edm::EDGetTokenT<pat::METCollection> met_token;
    edm::EDGetTokenT<edm::TriggerResults> metFiltersResults_token;
    std::map<std::string, edm::EDGetTokenT<bool>> customMetFilters_token;
    SelectionHist selection;
};

} // namespace tau_trigger

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTriggerSelectionFilter = tau_trigger::SelectionFilter;
DEFINE_FWK_MODULE(TauTriggerSelectionFilter);
