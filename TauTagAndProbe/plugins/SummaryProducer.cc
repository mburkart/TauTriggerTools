/*! Creates tuple for tau analysis.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#include "Compression.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "TauTriggerTools/Common/interface/GenTruthTools.h"
#include "TauTriggerTools/Common/interface/TriggerDescriptor.h"
#include "TauTriggerTools/TauTagAndProbe/interface/SummaryTuple.h"

namespace tau_trigger {

using namespace analysis;

class SummaryProducer : public edm::stream::EDProducer<edm::GlobalCache<SummaryProducerData>> {
public:
    SummaryProducer(const edm::ParameterSet& cfg, const SummaryProducerData* globalData) :
        isMC(cfg.getParameter<bool>("isMC")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        vertices_token(mayConsume<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("vertices"))),
        data(*globalData)
    {
        produces<bool>();
    }

    static std::unique_ptr<SummaryProducerData> initializeGlobalCache(const edm::ParameterSet& cfg)
    {
        TFile& file = edm::Service<TFileService>()->file();
        file.SetCompressionAlgorithm(ROOT::kLZ4);
        file.SetCompressionLevel(4);
        const bool isMC = cfg.getParameter<bool>("isMC");
        std::cout << "--A" << std::endl;
        auto data = std::make_unique<SummaryProducerData>(file, isMC);
        std::cout << "--B" << std::endl;
        SummaryTuple& summaryTuple = *data->getSummaryTuple();
        summaryTuple().numberOfProcessedEvents = 0;
        summaryTuple().totalGenEventWeight = 0;
        std::cout << "--C" << std::endl;
        TriggerDescriptorCollection hltPaths(cfg.getParameter<edm::VParameterSet>("hltPaths"));
        for(unsigned n = 0; n < hltPaths.size(); ++n) {
            summaryTuple().trigger_index.push_back(n);
            summaryTuple().trigger_pattern.push_back(hltPaths.at(n).path);
        }
        std::cout << "--D" << std::endl;
        return data;
    }

    static void globalEndJob(SummaryProducerData* data)
    {
        SummaryProducerData::LockGuard lock(data->getMutex());
        if(data->getExpressTuple())
            data->getExpressTuple()->Write();
        SummaryTuple& summaryTuple = *data->getSummaryTuple();
        const auto& filters = data->getFilters();
        for(const auto& entry : filters) {
            summaryTuple().filter_name.push_back(entry.first);
            summaryTuple().filter_hash.push_back(entry.second);
        }
        summaryTuple().exeTime = data->getElapsedTime();
        summaryTuple.Fill();
        summaryTuple.Write();
    }

private:
    static constexpr float default_value = ::tau_trigger::DefaultFillValue<float>();
    static constexpr int default_int_value = ::tau_trigger::DefaultFillValue<int>();

    virtual void produce(edm::Event& event, const edm::EventSetup&) override
    {
        event.put(std::make_unique<bool>(true));

        SummaryProducerData::LockGuard lock(data.getMutex());

        SummaryTuple& summaryTuple = *data.getSummaryTuple();
        summaryTuple().numberOfProcessedEvents++;

        float genWeight = default_value;
        int npu = default_int_value;
        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);
            genWeight = static_cast<float>(genEvent->weight());
            summaryTuple().totalGenEventWeight += genWeight;

            edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
            event.getByToken(puInfo_token, puInfo);
            npu = gen_truth::GetNumberOfPileUpInteractions(puInfo);
        }

        if(data.getExpressTuple()) {
            ExpressTuple& expressTuple = *data.getExpressTuple();
            expressTuple().run  = event.id().run();
            expressTuple().lumi = event.id().luminosityBlock();
            expressTuple().evt  = event.id().event();

            edm::Handle<std::vector<reco::Vertex>> vertices;
            event.getByToken(vertices_token, vertices);
            expressTuple().npv = static_cast<int>(vertices->size());
            expressTuple().genEventWeight = genWeight;
            expressTuple().npu = npu;

            expressTuple.Fill();
        }
    }

private:
    const bool isMC;

    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfo_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token;

    const SummaryProducerData& data;
};

} // namespace tau_trigger

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTriggerSummaryTupleProducer = tau_trigger::SummaryProducer;
DEFINE_FWK_MODULE(TauTriggerSummaryTupleProducer);
