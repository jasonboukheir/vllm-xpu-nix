// Enumerate Level Zero metric capabilities without submitting GPU work.
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static std::string quote(const char* raw) {
    std::string out = "\"";
    for (const unsigned char c : std::string(raw ? raw : "")) {
        if (c == '\\' || c == '"') out += '\\';
        if (c < 32) {
            const char* hex = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4]; out += hex[c & 15];
        } else out += c;
    }
    return out + '"';
}

static bool checked(ze_result_t status, const char* operation) {
    if (status == ZE_RESULT_SUCCESS) return true;
    std::cerr << operation << " failed with Level Zero status "
              << static_cast<unsigned>(status) << '\n';
    return false;
}

// Open and immediately close a stream: permission/capability check only.
// No workload is submitted and no collected values are claimed as valid.
static void check_stream(ze_driver_handle_t driver, ze_device_handle_t device,
                         zet_metric_group_handle_t group) {
    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t context = nullptr;
    auto status = zeContextCreate(driver, &context_desc, &context);
    std::cout << ",\"stream_check\":{\"context_create_status\":"
              << static_cast<unsigned>(status);
    if (status == ZE_RESULT_SUCCESS) {
        status = zetContextActivateMetricGroups(context, device, 1, &group);
        std::cout << ",\"activate_status\":" << static_cast<unsigned>(status);
        if (status == ZE_RESULT_SUCCESS) {
            zet_metric_streamer_desc_t desc{};
            desc.stype = ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC;
            desc.notifyEveryNReports = 1;
            desc.samplingPeriod = 100000; // 100 us, not a performance experiment.
            zet_metric_streamer_handle_t stream = nullptr;
            status = zetMetricStreamerOpen(context, device, group, &desc, nullptr, &stream);
            std::cout << ",\"open_status\":" << static_cast<unsigned>(status)
                      << ",\"sampling_period_ns\":" << desc.samplingPeriod;
            if (status == ZE_RESULT_SUCCESS) {
                std::cout << ",\"close_status\":"
                          << static_cast<unsigned>(zetMetricStreamerClose(stream));
            }
            std::cout << ",\"deactivate_status\":"
                      << static_cast<unsigned>(
                             zetContextActivateMetricGroups(context, device, 0, nullptr));
        }
        std::cout << ",\"context_destroy_status\":"
                  << static_cast<unsigned>(zeContextDestroy(context));
    }
    std::cout << '}';
}

int main(int argc, char** argv) {
    const std::string expected = argc > 1 ? argv[1] : "Intel(R) Arc(TM) Pro B70 Graphics";
    const std::string stream_group = argc > 2 ? argv[2] : "";
    if (!checked(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit")) return 1;
    uint32_t driver_count = 0;
    if (!checked(zeDriverGet(&driver_count, nullptr), "zeDriverGet")) return 1;
    std::vector<ze_driver_handle_t> drivers(driver_count);
    if (!checked(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet")) return 1;
    std::cout << "{\"schema_version\":1,\"collection_validated\":false,"
              << "\"metrics_enabled\":" << quote(std::getenv("ZET_ENABLE_METRICS"))
              << ",\"devices\":[";
    bool first = true, found = false, available = false;
    for (auto driver : drivers) {
        uint32_t count = 0;
        if (!checked(zeDeviceGet(driver, &count, nullptr), "zeDeviceGet")) return 1;
        std::vector<ze_device_handle_t> devices(count);
        if (!checked(zeDeviceGet(driver, &count, devices.data()), "zeDeviceGet")) return 1;
        for (auto device : devices) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (!checked(zeDeviceGetProperties(device, &props), "zeDeviceGetProperties")) return 1;
            if (!first) std::cout << ',';
            first = false;
            const bool target = expected == props.name;
            std::cout << "{\"name\":" << quote(props.name)
                      << ",\"is_target\":" << (target ? "true" : "false");
            if (target) {
                found = true;
                uint32_t group_count = 0;
                ze_result_t status = zetMetricGroupGet(device, &group_count, nullptr);
                std::vector<zet_metric_group_handle_t> groups(group_count);
                if (status == ZE_RESULT_SUCCESS && group_count)
                    status = zetMetricGroupGet(device, &group_count, groups.data());
                std::cout << ",\"metric_group_status\":" << static_cast<unsigned>(status)
                          << ",\"metric_groups\":[";
                if (status == ZE_RESULT_SUCCESS) {
                    for (uint32_t i = 0; i < group_count; ++i) {
                        zet_metric_group_properties_t group{};
                        group.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
                        auto group_status = zetMetricGroupGetProperties(groups[i], &group);
                        if (i) std::cout << ',';
                        std::cout << "{\"status\":" << static_cast<unsigned>(group_status);
                        if (group_status == ZE_RESULT_SUCCESS) {
                            available = true;
                            std::cout << ",\"name\":" << quote(group.name)
                                      << ",\"description\":" << quote(group.description)
                                      << ",\"sampling_type_flags\":" << group.samplingType
                                      << ",\"domain\":" << group.domain
                                      << ",\"metric_count\":" << group.metricCount;
                            uint32_t metric_count = 0;
                            auto metric_count_status =
                                zetMetricGet(groups[i], &metric_count, nullptr);
                            std::vector<zet_metric_handle_t> metrics(metric_count);
                            auto metric_get_status = ZE_RESULT_SUCCESS;
                            if (metric_count_status == ZE_RESULT_SUCCESS && metric_count)
                                metric_get_status =
                                    zetMetricGet(groups[i], &metric_count, metrics.data());
                            std::cout << ",\"metrics_count_status\":"
                                      << static_cast<unsigned>(metric_count_status)
                                      << ",\"metrics_get_status\":"
                                      << static_cast<unsigned>(metric_get_status)
                                      << ",\"metrics\":[";
                            if (metric_count_status == ZE_RESULT_SUCCESS &&
                                metric_get_status == ZE_RESULT_SUCCESS) {
                                for (uint32_t m = 0; m < metric_count; ++m) {
                                    zet_metric_properties_t metric{};
                                    metric.stype = ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
                                    auto properties_status =
                                        zetMetricGetProperties(metrics[m], &metric);
                                    if (m) std::cout << ',';
                                    std::cout << "{\"properties_status\":"
                                              << static_cast<unsigned>(properties_status);
                                    if (properties_status == ZE_RESULT_SUCCESS) {
                                        std::cout << ",\"name\":" << quote(metric.name)
                                                  << ",\"description\":"
                                                  << quote(metric.description)
                                                  << ",\"component\":"
                                                  << quote(metric.component)
                                                  << ",\"result_type\":"
                                                  << metric.resultType
                                                  << ",\"metric_type\":"
                                                  << metric.metricType
                                                  << ",\"units\":"
                                                  << quote(metric.resultUnits);
                                    }
                                    std::cout << '}';
                                }
                            }
                            std::cout << ']';
                            if (stream_group == group.name &&
                                (group.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED)) {
                                check_stream(driver, device, groups[i]);
                            }
                        }
                        std::cout << '}';
                    }
                }
                std::cout << ']';
            }
            std::cout << '}';
        }
    }
    std::cout << "],\"target_found\":" << (found ? "true" : "false")
              << ",\"metric_groups_available\":" << (available ? "true" : "false") << "}\n";
    return found ? 0 : 2;
}
