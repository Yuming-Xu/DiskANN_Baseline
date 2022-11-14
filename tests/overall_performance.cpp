#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <index.h>
#include <future>
#include <Neighbor_Tag.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define Merge_Size 1000000
#define NUM_INSERT_THREADS 16
#define NUM_SEARCH_THREADS 10

int            begin_time = 0;
diskann::Timer globalTimer;

// acutually also shows disk size
void ShowMemoryStatus() {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int           tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  std::cout << "memory current time: " << current_time << " RSS : " << rss
            << " KB" << std::endl;
  // char           dir[] = "/home/yuming/ssdfile/store_diskann_100m";
  // DIR*           dp;
  // struct dirent* entry;
  // struct stat    statbuf;
  // long           dir_size = 0;

  // if ((dp = opendir(dir)) == NULL) {
  //   fprintf(stderr, "Cannot open dir: %s\n", dir);
  //   exit(0);
  // }

  // chdir(dir);

  // while ((entry = readdir(dp)) != NULL) {
  //   lstat(entry->d_name, &statbuf);
  //   dir_size += statbuf.st_size;
  // }
  // chdir("..");
  // closedir(dp);
  // dir_size /= (1024 * 1024);
  // std::cout << "disk usage : " << dir_size << " MB" << std::endl;
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(std::string& truthFilePrefix, int vectorCount) {
  std::string fileName(truthFilePrefix);
  fileName += "-";
  if (vectorCount < 1000) {
    fileName += std::to_string(vectorCount);
  } else if (vectorCount < 1000000) {
    fileName += std::to_string(vectorCount / 1000);
    fileName += "k";
  } else if (vectorCount < 1000000000) {
    if (vectorCount % 1000000 == 0) {
      fileName += std::to_string(vectorCount / 1000000);
      fileName += "M";
    } else {
      float vectorCountM = ((float) vectorCount) / 1000000;
      fileName += convertFloatToString(vectorCountM, 2);
      fileName += "M";
    }
  } else {
    fileName += std::to_string(vectorCount / 1000000000);
    fileName += "B";
  }
  return fileName;
}

template<typename T, typename TagT>
void sync_search_kernel(T* query, size_t query_num, size_t query_aligned_dim,
                        const int recall_at, _u64 L,
                        diskann::MergeInsert<T, TagT>& sync_index,
                        std::string&                   truthset_file,
                        tsl::robin_set<TagT>& inactive_tags, int curCount,
                        bool merged, bool calRecall) {
  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;

  if (calRecall) {
    std::cout << "current truthfile: " << truthset_file << std::endl;
    // diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float* query_result_dists = new float[recall_at * query_num];
  TagT*  query_result_tags = new TagT[recall_at * query_num];

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double>  latency_stats(query_num, 0);
  diskann::QueryStats* stats = new diskann::QueryStats[query_num];
  std::string          recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Latency (ms)" << std::setw(12) << "90 Latency"
            << std::setw(12) << "95 Latency" << std::setw(12) << "99 Latency"
            << std::setw(12) << "99.9 Latency" << std::setw(12) << recall_string
            << std::setw(12) << "Mean disk IOs" << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();
    sync_index.search_sync(query + i * query_aligned_dim, recall_at, L,
                           query_result_tags + i * recall_at,
                           query_result_dists + i * recall_at, stats + i);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float                         qps = (query_num / diff.count());
  float                         recall = 0;

  if (calRecall) {
    if (merged) {
      std::string cur_result_path =
          "/home/yuming/_" + std::to_string(curCount) + "merged.bin";
      diskann::save_bin<TagT>(cur_result_path, query_result_tags, query_num,
                              recall_at);
    } else {
      std::string cur_result_path =
          "/home/yuming/_" + std::to_string(curCount) + ".bin";
      diskann::save_bin<TagT>(cur_result_path, query_result_tags, query_num,
                              recall_at);
    }

    // recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
    //                                    query_result_tags, recall_at,
    //                                    recall_at, inactive_tags);
  }

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  std::cout << "search current time: " << current_time << std::endl;

  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats& stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(),
                                        latency_stats.end(), 0)) /
                   (float) query_num
            << std::setw(12)
            << (float) latency_stats[(_u64)(0.90 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64)(0.95 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64)(0.99 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64)(0.999 * ((double) query_num))]
            << std::setw(12) << recall << std::setw(12) << mean_ios
            << std::endl;

  delete[] query_result_dists;
  delete[] query_result_tags;
}

template<typename T, typename TagT>
void merge_kernel(diskann::MergeInsert<T, TagT>& sync_index,
                  std::string&                   save_path) {
  sync_index.final_merge();
}

template<typename T, typename TagT>
void insertion_kernel(T* data_load, diskann::MergeInsert<T, TagT>& sync_index,
                      std::vector<TagT>& insert_vec, size_t aligned_dim) {
  diskann::Timer timer;

  std::cout << "Begin Insert" << std::endl;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
    sync_index.insert(data_load + aligned_dim * insert_vec[i], insert_vec[i]);
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::cout << "Inserted " << insert_vec.size() << " points in " << time_secs
            << "s" << std::endl;
}

template<typename T, typename TagT>
void update(const std::string& data_path, const unsigned L_mem,
            const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
            const unsigned R_disk, const float alpha_disk, int step,
            const size_t base_num, const unsigned num_pq_chunks,
            const unsigned nodes_to_cache, std::string& save_path,
            const std::string& query_file, std::string& truthset_file,
            const int recall_at, _u64 Lsearch, const unsigned beam_width,
            diskann::Distance<T>* dist_cmp) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 160);
  paras.Set<unsigned>("beamwidth", beam_width);
  // paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_search_threads", 32);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::Timer timer;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  std::cout << "Loaded full data for driver: (" << num_points << "," << dim
            << ") vectors." << std::endl;
  diskann::Metric metric = diskann::Metric::L2;
  // diskann::SyncIndex<T, TagT> sync_index(num_points + 5000, dim, num_shards,
  //                                        paras, NUM_SEARCH_THREADS,
  //                                        save_path);
  // size_t                      res = sync_index.load(save_path);
  diskann::MergeInsert<T, TagT> sync_index(paras, dim, save_path + "_mem",
                                           save_path, save_path + "_merge",
                                           dist_cmp, metric, false, save_path);

  std::cout << "Loading queries " << std::endl;
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  std::cout << "Searching before inserts: " << std::endl;

  std::string currentFileName = GetTruthFileName(truthset_file, base_num);

  tsl::robin_set<TagT> inactive_tags;

  for (_u64 j = base_num; j < num_points; j++) {
    inactive_tags.insert(j);
  }

  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus();

  sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch,
                     sync_index, currentFileName, inactive_tags, base_num,
                     false, true);

  int               batch = 0;
  int               inMmeorySize = 0;
  int               res = base_num;
  std::future<void> merge_future;
  if (step != 0)
    batch = (num_points - base_num) / step;

  for (int i = 0; i < batch; i++) {
    std::cout << "batch: " << i << " step : " << step << std::endl;
    std::vector<unsigned> insert_vec;
    for (unsigned j = res; j < (res + step); j++)
      insert_vec.push_back(j);

    std::future<void> insert_future =
        std::async(std::launch::async, insertion_kernel<T, TagT>, data_load,
                   std::ref(sync_index), std::ref(insert_vec), aligned_dim);

    int                total_queries = 0;
    std::future_status insert_status;
    do {
      insert_status = insert_future.wait_for(std::chrono::milliseconds(1));
      if (insert_status == std::future_status::deferred) {
        std::cout << "deferred\n";
      } else if (insert_status == std::future_status::timeout) {
        ShowMemoryStatus();
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                           Lsearch, sync_index, currentFileName, inactive_tags,
                           res, false, false);
        total_queries += query_num;
        std::cout << "Queries processed: " << total_queries << std::endl;
      }
      if (insert_status == std::future_status::ready) {
        std::cout << "Insertions complete!\n";
      }
    } while (insert_status != std::future_status::ready);

    res += step;
    inMmeorySize += step;

    std::cout << "Search after inserts, current vector number: " << res
              << std::endl;

    currentFileName.clear();

    currentFileName = GetTruthFileName(truthset_file, res);

    inactive_tags.clear();
    for (_u64 j = res; j < num_points; j++) {
      inactive_tags.insert(j);
    }

    sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch,
                       sync_index, currentFileName, inactive_tags, res, false,
                       true);

    if (i == batch - 1) {
      std::cout << "Begin Merge" << std::endl;
      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>,
                                std::ref(sync_index), std::ref(save_path));
      std::future_status merge_status;
      total_queries = 0;
      int second = 0;
      do {
        merge_status = merge_future.wait_for(std::chrono::seconds(1));
        ShowMemoryStatus();
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                           Lsearch, sync_index, currentFileName, inactive_tags,
                           res, false, false);
        total_queries += query_num;
        std::cout << "Queries processed: " << total_queries << std::endl;
      } while (merge_status != std::future_status::ready);
      std::cout << "Merge complete. Now searching..." << std::endl;

      inMmeorySize = 0;

      currentFileName.clear();

      currentFileName = GetTruthFileName(truthset_file, res);

      inactive_tags.clear();
      for (_u64 j = res; j < num_points; j++) {
        inactive_tags.insert(j);
      }
      sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                         Lsearch, sync_index, currentFileName, inactive_tags,
                         res, true, true);
    } else if (inMmeorySize >= Merge_Size) {
      std::cout << "Begin Merge" << std::endl;
      // merge_future = std::async(std::launch::async, merge_kernel<T, TagT>,
      //                           std::ref(sync_index), std::ref(save_path));
      // std::this_thread::sleep_for(std::chrono::seconds(5));
      merge_kernel<T, TagT>(sync_index, save_path);
      std::cout << "Sending Merge" << std::endl;
      inMmeorySize = 0;
    }
  }
  delete[] data_load;
}

template<typename T, typename TagT>
void build(const std::string& data_path, const unsigned L_mem,
           const unsigned R_mem, const float alpha_mem, const unsigned L_disk,
           const unsigned R_disk, const float alpha_disk,
           const size_t num_start, const size_t num_shards,
           const unsigned num_pq_chunks, const unsigned nodes_to_cache,
           const std::string& save_path) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", L_mem);
  paras.Set<unsigned>("R_mem", R_mem);
  paras.Set<float>("alpha_mem", alpha_mem);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 500);
  paras.Set<unsigned>("beamwidth", 5);
  paras.Set<unsigned>("num_pq_chunks", num_pq_chunks);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  std::cout << "Loaded full data for driver." << std::endl;
  // diskann::SyncIndex<T, TagT> sync_index(num_points + 5000, dim, num_shards,
  //                                        paras, 2, save_path);
  std::cout << "Ran constructor." << std::endl;
  std::vector<TagT> tags(num_start);
  std::iota(tags.begin(), tags.end(), 0);
  diskann::Timer timer;
  // sync_index.build(data_path.c_str(), num_start, tags);
  std::cout << "Sync Index build time: " << timer.elapsed() / 1000000 << "s\n";

  std::string tag_file = save_path + "_disk.index.tags";

  diskann::save_bin<TagT>(tag_file, tags.data(), num_start, 1);
  delete[] data_load;
}

int main(int argc, char** argv) {
  if (argc < 14) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <base_data_file> <L_mem> <R_mem> "
                 "<alpha_mem>"
              << " <L_disk> <R_disk> <alpha_disk>"
              << " <num_start> <num_shards> <#pq_chunks> <#nodes_to_cache>"
              << " <save_graph_file> <update> <build> <full_data_file> "
                 "<query_file> <truthset_file> <recall@>"
              << " <Lsearch> <#beam_width>"
              << " <step>" << std::endl;
    exit(-1);
  }

  int         arg_no = 3;
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) std::atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) std::atof(argv[arg_no++]);
  size_t      num_start = (size_t) std::atoi(argv[arg_no++]);
  size_t      num_shards = (size_t) std::atoi(argv[arg_no++]);
  unsigned    num_pq_chunks = (unsigned) std::atoi(argv[arg_no++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);
  bool        updateIndex = false;
  bool        buildIndex = false;

  if (std::string(argv[arg_no++]) == std::string("true"))
    updateIndex = true;
  if (std::string(argv[arg_no++]) == std::string("true"))
    buildIndex = true;

  std::string full_data_path(argv[arg_no++]);
  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int         recall_at = (int) std::atoi(argv[arg_no++]);
  _u64        Lsearch = std::atoi(argv[arg_no++]);
  unsigned    beam_width = (unsigned) std::atoi(argv[arg_no++]);
  int         step = (int) std::atoi(argv[arg_no++]);

  if (!updateIndex || buildIndex) {
    if (std::string(argv[1]) == std::string("int8"))
      build<int8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                              alpha_disk, num_start, num_shards, num_pq_chunks,
                              nodes_to_cache, save_path);
    else if (std::string(argv[1]) == std::string("uint8"))
      build<uint8_t, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                               alpha_disk, num_start, num_shards, num_pq_chunks,
                               nodes_to_cache, save_path);
    else if (std::string(argv[1]) == std::string("float"))
      build<float, unsigned>(argv[2], L_mem, R_mem, alpha_mem, L_disk, R_disk,
                             alpha_disk, num_start, num_shards, num_pq_chunks,
                             nodes_to_cache, save_path);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  }

  if (!updateIndex)
    return 0;

  if (updateIndex) {
    if (std::string(argv[1]) == std::string("int8")) {
      diskann::DistanceL2Int8 dist_cmp;
      update<int8_t, unsigned>(
          full_data_path, L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
          step, num_start, num_pq_chunks, nodes_to_cache, save_path, query_file,
          truthset, recall_at, Lsearch, beam_width, &dist_cmp);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      diskann::DistanceL2UInt8 dist_cmp;
      update<uint8_t, unsigned>(
          full_data_path, L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
          step, num_start, num_pq_chunks, nodes_to_cache, save_path, query_file,
          truthset, recall_at, Lsearch, beam_width, &dist_cmp);
    } else if (std::string(argv[1]) == std::string("float")) {
      diskann::DistanceL2 dist_cmp;
      update<float, unsigned>(
          full_data_path, L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
          step, num_start, num_pq_chunks, nodes_to_cache, save_path, query_file,
          truthset, recall_at, Lsearch, beam_width, &dist_cmp);
    } else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  }
}