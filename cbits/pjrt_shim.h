#ifndef PJRT_SHIM_H
#define PJRT_SHIM_H

#include "pjrt_c_api.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Plugin loading
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_load_plugin(const char* path, PJRT_Api** out_api);

/* ---------------------------------------------------------------------------
 * Client
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_create_client(PJRT_Api* api, PJRT_Client** out_client);
PJRT_Error* hhlo_pjrt_client_destroy(PJRT_Api* api, PJRT_Client* client);

/* ---------------------------------------------------------------------------
 * Device enumeration
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_client_addressable_device_count(PJRT_Api* api,
                                                       PJRT_Client* client,
                                                       size_t* out_count);
PJRT_Error* hhlo_pjrt_client_addressable_device(PJRT_Api* api,
                                                 PJRT_Client* client,
                                                 size_t index,
                                                 PJRT_Device** out_device);
PJRT_Error* hhlo_pjrt_device_id(PJRT_Api* api, PJRT_Device* device, int* out_id);
PJRT_Error* hhlo_pjrt_device_kind(PJRT_Api* api, PJRT_Device* device,
                                   const char** out_kind, size_t* out_kind_len);

/* ---------------------------------------------------------------------------
 * Compilation
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_compile(PJRT_Api* api, PJRT_Client* client,
                               const char* code, size_t code_size,
                               PJRT_LoadedExecutable** out_exec);
PJRT_Error* hhlo_pjrt_loaded_executable_destroy(PJRT_Api* api,
                                                 PJRT_LoadedExecutable* exec);
PJRT_Error* hhlo_pjrt_executable_num_outputs(PJRT_Api* api,
                                              PJRT_LoadedExecutable* loaded_exec,
                                              size_t* out_num_outputs);

/* ---------------------------------------------------------------------------
 * Execution
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_execute(PJRT_Api* api, PJRT_LoadedExecutable* exec,
                               size_t num_args, PJRT_Buffer** args_in,
                               size_t max_outputs,
                               PJRT_Buffer** out_outputs,
                               size_t* out_num_outputs);

PJRT_Error* hhlo_pjrt_execute_on_device(PJRT_Api* api,
                                         PJRT_LoadedExecutable* exec,
                                         size_t num_args, PJRT_Buffer** args_in,
                                         PJRT_Device* execute_device,
                                         size_t max_outputs,
                                         PJRT_Buffer** out_outputs,
                                         size_t* out_num_outputs);

/* ---------------------------------------------------------------------------
 * Buffers
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_buffer_from_host(PJRT_Api* api, PJRT_Client* client,
                                        const void* data,
                                        PJRT_Buffer_Type type,
                                        const int64_t* dims, size_t num_dims,
                                        PJRT_Buffer** out_buffer);

PJRT_Error* hhlo_pjrt_buffer_from_host_on_device(PJRT_Api* api,
                                                  PJRT_Client* client,
                                                  PJRT_Device* device,
                                                  const void* data,
                                                  PJRT_Buffer_Type type,
                                                  const int64_t* dims,
                                                  size_t num_dims,
                                                  PJRT_Buffer** out_buffer);

PJRT_Error* hhlo_pjrt_buffer_to_host(PJRT_Api* api, PJRT_Buffer* buffer,
                                      void* dst, size_t dst_size,
                                      PJRT_Event** out_event);

PJRT_Error* hhlo_pjrt_buffer_to_host_async(PJRT_Api* api, PJRT_Buffer* buffer,
                                            void* dst, size_t dst_size,
                                            PJRT_Event** out_event);

PJRT_Error* hhlo_pjrt_buffer_destroy(PJRT_Api* api, PJRT_Buffer* buffer);
PJRT_Error* hhlo_pjrt_buffer_dimensions(PJRT_Api* api, PJRT_Buffer* buffer,
                                         const int64_t** out_dims,
                                         size_t* out_num_dims);
PJRT_Error* hhlo_pjrt_buffer_element_type(PJRT_Api* api, PJRT_Buffer* buffer,
                                           int* out_type);
PJRT_Error* hhlo_pjrt_buffer_on_device_size(PJRT_Api* api, PJRT_Buffer* buffer,
                                             size_t* out_size);

/* ---------------------------------------------------------------------------
 * Events
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_buffer_ready_event(PJRT_Api* api, PJRT_Buffer* buffer,
                                          PJRT_Event** out_event);
PJRT_Error* hhlo_pjrt_event_is_ready(PJRT_Api* api, PJRT_Event* event,
                                      int* out_ready);
PJRT_Error* hhlo_pjrt_event_await(PJRT_Api* api, PJRT_Event* event);
PJRT_Error* hhlo_pjrt_event_destroy(PJRT_Api* api, PJRT_Event* event);

/* ---------------------------------------------------------------------------
 * Errors
 * --------------------------------------------------------------------------- */
PJRT_Error* hhlo_pjrt_error_message(PJRT_Api* api, PJRT_Error* error,
                                     const char** out_msg, size_t* out_size);
PJRT_Error* hhlo_pjrt_error_destroy(PJRT_Api* api, PJRT_Error* error);

/* ---------------------------------------------------------------------------
 * Buffer type constants
 * --------------------------------------------------------------------------- */
int hhlo_buffer_type_invalid(void);
int hhlo_buffer_type_pred(void);
int hhlo_buffer_type_s8(void);
int hhlo_buffer_type_s16(void);
int hhlo_buffer_type_s32(void);
int hhlo_buffer_type_s64(void);
int hhlo_buffer_type_u8(void);
int hhlo_buffer_type_u16(void);
int hhlo_buffer_type_u32(void);
int hhlo_buffer_type_u64(void);
int hhlo_buffer_type_f16(void);
int hhlo_buffer_type_f32(void);
int hhlo_buffer_type_f64(void);
int hhlo_buffer_type_bf16(void);
int hhlo_buffer_type_c64(void);
int hhlo_buffer_type_c128(void);

#ifdef __cplusplus
}
#endif

#endif /* PJRT_SHIM_H */
