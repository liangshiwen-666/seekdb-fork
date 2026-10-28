/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OCEANBASE_BASIC_OB_SCALAR_GROUPBY_OP_H_
#define OCEANBASE_BASIC_OB_SCALAR_GROUPBY_OP_H_

#include <atomic>
#include <mutex>

#include "common/row/ob_row_store.h"
#include "sql/engine/aggregate/ob_groupby_op.h"
#include "sql/engine/basic/ob_hp_infrastructure_manager.h"

namespace oceanbase
{
namespace sql
{

// Global data version used to invalidate cached scalar aggregate results. It is
// bumped whenever a statement that may modify data (DML/DDL) completes, so a
// cached result is reused only while no write has happened in between.
inline std::atomic<int64_t> &query_result_cache_version()
{
  static std::atomic<int64_t> version(0);
  return version;
}

inline int64_t get_query_result_cache_version()
{
  return query_result_cache_version().load(std::memory_order_relaxed);
}

inline void bump_query_result_cache_version()
{
  query_result_cache_version().fetch_add(1, std::memory_order_relaxed);
}

// Caches the single output row of a scalar aggregate operator. The cache is a
// member of the operator spec (which lives as long as the cached physical
// plan), so repeated executions of the same plan can skip the child operator
// tree while the data is unchanged. Only plain COUNT aggregates are cached.
class ObScalarAggResultCache
{
public:
  ObScalarAggResultCache() : version_(-1), param_hash_(0), valid_(false), count_(0) {}
  bool hit(const int64_t version, const uint64_t param_hash) const
  {
    return valid_ && version_ == version && param_hash_ == param_hash;
  }
  int64_t get_count() const { return count_; }
  void store(const int64_t version, const uint64_t param_hash, const int64_t count)
  {
    std::lock_guard<std::mutex> guard(lock_);
    version_ = version;
    param_hash_ = param_hash;
    count_ = count;
    valid_ = true;
  }
  void invalidate() { valid_ = false; }
private:
  std::mutex lock_;
  int64_t version_;
  uint64_t param_hash_;
  bool valid_;
  int64_t count_;
};

class ObScalarAggregateSpec : public ObGroupBySpec
{
  OB_UNIS_VERSION_V(1);
public:
  ObScalarAggregateSpec(common::ObIAllocator &alloc, const ObPhyOperatorType type)
    : ObGroupBySpec(alloc, type), enable_hash_base_distinct_(false)
  {}

public:
  bool enable_hash_base_distinct_;
  // Runtime only result cache, not serialized with the plan.
  mutable ObScalarAggResultCache result_cache_;

private:
  // disallow copy
  DISALLOW_COPY_AND_ASSIGN(ObScalarAggregateSpec);
};

class ObScalarAggregateOp : public ObGroupByOp
{
public:
  friend ObAggregateProcessor;
  ObScalarAggregateOp(ObExecContext &exec_ctx, const ObOpSpec &spec, ObOpInput *input)
    : ObGroupByOp(exec_ctx, spec, input), started_(false), dir_id_(-1),
      profile_(ObSqlWorkAreaType::HASH_WORK_AREA),
      sql_mem_processor_(profile_, op_monitor_info_),
      hp_infras_mgr_()
  {
  }

  virtual int inner_open() override;
  virtual int inner_close() override;
  virtual int inner_rescan() override;
  virtual int inner_switch_iterator() override;
  virtual int inner_get_next_row() override;
  virtual int inner_get_next_batch(const int64_t max_row_cnt) override;
  virtual void destroy() override;
  // reset default value of %cur_rownum_ && %rownum_limit_
private:
  int init_hp_infras_group_mgr();
  uint64_t calc_param_hash() const;
  bool cacheable_count() const;
  int restore_cached_count();
  int save_cached_count(const int64_t version, const uint64_t param_hash);
private:
  // disallow copy
  DISALLOW_COPY_AND_ASSIGN(ObScalarAggregateOp);

private:
  bool started_;
  int64_t dir_id_;
  ObSqlWorkAreaProfile profile_;
  ObSqlMemMgrProcessor sql_mem_processor_;
  HashPartInfrasMgr hp_infras_mgr_;
};

} // end namespace sql
} // end namespace oceanbase

#endif // OCEANBASE_BASIC_OB_SCALAR_GROUPBY_OP_H_
