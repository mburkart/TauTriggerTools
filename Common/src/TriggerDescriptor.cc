/*! Definition of trigger results.
This file is part of https://github.com/cms-tau-pog/TauTriggerTools. */

#include "TauTriggerTools/Common/interface/TriggerDescriptor.h"
#include <Math/VectorUtil.h>
#include "TauTriggerTools/Common/interface/TextIO.h"

namespace tau_trigger {
unsigned GetTriggerObjectTypes(const pat::TriggerObjectStandAlone& triggerObject)
{
    unsigned type = 0;
    if(triggerObject.hasTriggerObjectType(trigger::TriggerElectron))
        type |= static_cast<unsigned>(analysis::LegType::e);
    if(triggerObject.hasTriggerObjectType(trigger::TriggerMuon))
        type |= static_cast<unsigned>(analysis::LegType::mu);
    if(triggerObject.hasTriggerObjectType(trigger::TriggerTau))
        type |= static_cast<unsigned>(analysis::LegType::tau);
    if(triggerObject.hasTriggerObjectType(trigger::TriggerJet))
        type |= static_cast<unsigned>(analysis::LegType::jet);
    return type;
}

const l1t::Tau* MatchL1Taus(const analysis::LorentzVectorM& ref_p4, const BXVector<l1t::Tau>& l1Taus, double deltaR2Thr,
                           int bx_value)
{
    const l1t::Tau* matched_tau = nullptr;
    double dR2_bestMatch = deltaR2Thr;
    for(auto iter = l1Taus.begin(0); iter != l1Taus.end(0); ++iter) {
        const double deltaR2 = ROOT::Math::VectorUtil::DeltaR2(ref_p4, iter->polarP4());
        if(deltaR2 < dR2_bestMatch) {
            matched_tau = &(*iter);
            dR2_bestMatch = deltaR2;
        }
    }
    return matched_tau;
}

TriggerDescriptorCollection::TriggerDescriptorCollection(const edm::VParameterSet& trig_vpset)
{
    if(trig_vpset.size() > MaxNumberOfTriggers)
        throw analysis::exception("The max number of triggers is exceeded");
    for(const auto& pset : trig_vpset) {
        TriggerDescriptor desc;
        desc.path = pset.getParameter<std::string>("path");
        desc.is_tag = pset.getParameter<bool>("is_tag");
        const std::vector<std::string> leg_types = pset.getParameter<std::vector<std::string>>("leg_types");
        const std::vector<std::string> filters = pset.getParameter<std::vector<std::string>>("filters");
        if(desc_indices.count(desc.path))
            throw analysis::exception("Duplicated trigger path = '%1%'.") % desc.path;
        if(leg_types.size() != filters.size())
            throw analysis::exception("Inconsitent leg_types and filters for trigger path = '%1%'.") % desc.path;

        static const std::string regex_format = "^%1%[0-9]+$";
        const std::string regex_str = boost::str(boost::format(regex_format) % desc.path);
        desc.regex = boost::regex(regex_str);
        desc.type_mask = 0;
        for(size_t n = 0; n < leg_types.size(); ++n) {
            TriggerLeg leg;
            leg.type = analysis::Parse<analysis::LegType>(leg_types.at(n));
            leg.filters = analysis::SplitValueList(filters.at(n), false);
            desc.type_mask |= static_cast<unsigned>(leg.type);
            desc.legs.push_back(leg);
        }
        const size_t desc_index = descs.size();
        descs.push_back(desc);
        desc_indices[desc.path] = desc_index;
        if(desc.is_tag)
            tag_desc_indices.insert(desc_index);
    }
}

FullTriggerResults TriggerDescriptorCollection::matchTriggerObjects(
        const edm::TriggerResults& triggerResults, const pat::TriggerObjectStandAloneCollection& triggerObjects,
        const analysis::LorentzVectorM& ref_p4, const std::vector<std::string>& triggerNames, double deltaR2Thr,
        bool include_tag_paths, bool include_nontag_paths)
{
    FullTriggerResults results;

    std::vector<unsigned> obj_types;
    for(const auto& hlt_obj : triggerObjects) {
        const unsigned obj_type = GetTriggerObjectTypes(hlt_obj);
        obj_types.push_back(obj_type);
    }

    for(size_t desc_index = 0; desc_index < descs.size(); ++desc_index) {
        const auto& trig_desc = descs.at(desc_index);
        if(!include_tag_paths && trig_desc.is_tag) continue;
        if(!include_nontag_paths && !trig_desc.is_tag) continue;
        if(trig_desc.global_index < 0) continue;
        const bool accept = triggerResults.accept(trig_desc.global_index);
        results.accept.set(desc_index, accept);
        const std::string& path_name = triggerNames.at(trig_desc.global_index);
        TriggerObjectMatchResult best_match;
        boost::optional<size_t> best_matched_obj_index;
        double dR2_bestMatch = deltaR2Thr;
        for(size_t obj_index = 0; obj_index < triggerObjects.size(); ++obj_index) {
            const auto& hlt_obj = triggerObjects.at(obj_index);
            if((obj_types.at(obj_index) & trig_desc.type_mask) == 0) continue;
            const double deltaR2 = ROOT::Math::VectorUtil::DeltaR2(ref_p4, hlt_obj.polarP4());
            if(deltaR2 >= deltaR2Thr) continue;
            if(!hlt_obj.hasPathName(path_name, true, false)) continue;
            if(deltaR2 < dR2_bestMatch) {
                best_matched_obj_index = obj_index;
                dR2_bestMatch = deltaR2;
            }
            results.acceptAndMatch.set(desc_index);
            auto& match_result = results.matchResults[obj_index];
            match_result.hltObjIndex = obj_index;
            match_result.objType = obj_types.at(obj_index);
            match_result.hasPathName.set(desc_index);
            match_result.descIndices.insert(desc_index);
            match_result.filters = hlt_obj.filterLabels();
        }

        if(best_matched_obj_index)
            results.matchResults[*best_matched_obj_index].isBestMatch.set(desc_index);
    }

    return results;
}

void TriggerDescriptorCollection::updateGlobalIndices(const std::vector<std::string>& triggerNames)
{
    std::map<size_t, size_t> globalToPos, posToGlobal;
    for(size_t pos = 0; pos < descs.size(); ++pos) {
        descs.at(pos).global_index = -1;
        for(size_t global_index = 0; global_index < triggerNames.size(); ++global_index) {
            if(boost::regex_match(triggerNames.at(global_index), descs.at(pos).regex)) {
                if(globalToPos.count(global_index)) {
                    throw analysis::exception("Trigger '%1%' matches with two path patterns: '%2%' and '%3%'.")
                        % triggerNames.at(global_index) % descs.at(globalToPos.at(global_index)).path
                        % descs.at(pos).path;
                }
                if(posToGlobal.count(pos)) {
                    throw analysis::exception("Path pattern '%1%' matches with two triggers: '%2' and '%3%'.")
                        % descs.at(pos).path % triggerNames.at(posToGlobal.at(pos)) % triggerNames.at(global_index);
                }
                globalToPos[global_index] = pos;
                posToGlobal[pos] = global_index;
                descs.at(pos).global_index = global_index;
            }
        }
    }
}

} // namespace tau_trigger
