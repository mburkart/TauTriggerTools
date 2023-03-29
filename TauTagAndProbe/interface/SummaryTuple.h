/*! Definition of a tuple with summary information about production.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#pragma once

#include "EventTuple.h"
#include "TauTriggerTools/Common/interface/Tools.h"

#define SUMMARY_DATA() \
    /* Run statistics */ \
    VAR(UInt_t, exeTime) \
    VAR(ULong64_t, numberOfProcessedEvents) \
    VAR(Double_t, totalGenEventWeight) \
    /* Trigger information */ \
    VAR(std::vector<UInt_t>, trigger_index) \
    VAR(std::vector<std::string>, trigger_pattern) \
    /* Filter information */ \
    VAR(std::vector<std::string>, filter_name) \
    VAR(std::vector<UInt_t>, filter_hash) \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_trigger, ProdSummary, SummaryTuple, SUMMARY_DATA, "summary")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_trigger, SummaryTuple, SUMMARY_DATA)
#undef VAR
#undef SUMMARY_DATA


#define EVENT_EXPRESS_DATA() \
    VAR(UInt_t, run) /* run */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Int_t, npv) /* number of primary vertices */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Float_t, npu) /* Number of in-time pu interactions added to the event */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_trigger, ExpressEvent, ExpressTuple, EVENT_EXPRESS_DATA, "all_events")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_trigger, ExpressTuple, EVENT_EXPRESS_DATA)
#undef VAR
#undef EVENT_EXPRESS_DATA

namespace tau_trigger {

struct SummaryProducerData {
public:
    using Mutex = SummaryTuple::Mutex;
    using LockGuard = std::lock_guard<Mutex>;
    using clock = std::chrono::system_clock;

    static SummaryProducerData& GetData()
    {
        if(GetDataPtr() == nullptr)
            throw analysis::exception("SummaryProducerData is not initialized.");
        return *GetDataPtr();
    }

private:
    static SummaryProducerData*& GetDataPtr()
    {
        static SummaryProducerData* data = nullptr;
        return data;
    }

public:
    SummaryProducerData(TFile& file, bool createExpressTuple) :
        start(clock::now()), summaryTuple(std::make_unique<SummaryTuple>("summary", &file, false))
    {
        if(createExpressTuple)
            expressTuple = std::make_unique<ExpressTuple>("all_events", &file, false);
        if(GetDataPtr() != nullptr)
            throw analysis::exception("Having multiple instances of SummaryProducerData is not supported.");
        GetDataPtr() = this;
    }

    SummaryTuple* getSummaryTuple() const { return summaryTuple.get(); }
    ExpressTuple* getExpressTuple() const { return expressTuple.get(); }
    Mutex& getMutex() const { return summaryTuple->GetMutex(); }

    unsigned getElapsedTime() const
    {
        return std::chrono::duration_cast<std::chrono::seconds>(clock::now() - start).count();
    }

    uint32_t getFilterHash(const std::string& filterName)
    {
        LockGuard lock(getMutex());
        auto iter = filterNameToHash.find(filterName);
        if(iter == filterNameToHash.end()) {
            const uint32_t hash = analysis::tools::hash(filterName);
            if(filterHashToName.count(hash))
                throw analysis::exception("Duplicated hash = %1% for filters '%2%' and '%3%'.") % hash % filterName
                        % filterHashToName.at(hash);
            filterNameToHash[filterName] = hash;
            filterHashToName[hash] = filterName;
            iter = filterNameToHash.find(filterName);
        }
        return iter->second;
    }

    const std::map<std::string, uint32_t>& getFilters() const { return filterNameToHash; }

private:
    const clock::time_point start;
    std::unique_ptr<SummaryTuple> summaryTuple;
    std::unique_ptr<ExpressTuple> expressTuple;
    std::map<std::string, uint32_t> filterNameToHash;
    std::map<uint32_t, std::string> filterHashToName;
};

}