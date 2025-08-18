use std::io::{BufReader, Cursor};

use wgpu::util::DeviceExt;

use crate::{model, texture};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    #[cfg(target_arch = "wasm32")]
    let txt = {
        let url = format_url(file_name);
        reqwest::get(url).await?.text().await?
    };
    #[cfg(not(target_arch = "wasm32"))]
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read_to_string(path)?
    };

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    let data = {
        let url = format_url(file_name);
        reqwest::get(url).await?.bytes().await?.to_vec()
    };
    #[cfg(not(target_arch = "wasm32"))]
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = std::io::Cursor::new(obj_text);
    let mut obj_reader = std::io::BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            match load_string(&p).await {
                Ok(mat_text) => {
                    log::info!("Loaded material file: {}", p);
                    tobj::load_mtl_buf(&mut std::io::BufReader::new(std::io::Cursor::new(mat_text)))
                }
                Err(e) => {
                    log::warn!("Failed to load material file '{}': {}", p, e);
                    Ok((Vec::new(), std::collections::HashMap::new())) // Return empty materials and map instead of failing
                }
            }
        },
    )
    .await?;

    // CRITICAL: Always ensure we have at least one material
    let mut materials = Vec::new();
    
    // Create default material first
    let default_material = model::Material {
        name: "default".to_string(),
        diffuse_texture: create_fallback_texture(device, queue)?,
        bind_group: create_material_bind_group(device, &create_fallback_texture(device, queue)?, layout, "default"),
    };
    materials.push(default_material);

    // Process loaded materials if any
    if let Ok(mtl_materials) = obj_materials {
        for (i, m) in mtl_materials.iter().enumerate() {
            log::info!("Processing material {}: '{}'", i, m.name);
            
            let texture = if m.diffuse_texture.is_empty() {
                log::info!("Material '{}' has no diffuse texture, using default", m.name);
                create_fallback_texture(device, queue)?
            } else {
                match load_texture(&m.diffuse_texture, device, queue).await {
                    Ok(tex) => {
                        log::info!("Loaded texture: {}", m.diffuse_texture);
                        tex
                    }
                    Err(e) => {
                        log::warn!("Failed to load texture '{}': {}. Using default.", m.diffuse_texture, e);
                        create_fallback_texture(device, queue)?
                    }
                }
            };

            materials.push(model::Material {
                name: m.name.clone(),
                diffuse_texture: texture,
                bind_group: create_material_bind_group(device, &materials.last().unwrap().diffuse_texture, layout, &m.name),
            });
        }
    } else {
        log::info!("No materials found in file, using default material only");
    }

    // Process meshes and ensure valid material indices
    let meshes = models
        .into_iter()
        .map(|m| {
            // CRITICAL: Clamp material index to valid range
            let material_index = if let Some(mat_id) = m.mesh.material_id {
                // Offset by 1 because we added default material at index 0
                let adjusted_index = mat_id + 1;
                if adjusted_index < materials.len() {
                    adjusted_index
                } else {
                    log::warn!("Mesh '{}' references invalid material index {}. Using default (0).", m.name, mat_id);
                    0 // Use default material
                }
            } else {
                log::info!("Mesh '{}' has no material assignment. Using default (0).", m.name);
                0 // Use default material
            };

            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: if i * 2 + 1 < m.mesh.texcoords.len() {
                        [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]]
                    } else {
                        [0.0, 0.0]
                    },
                    normal: if i * 3 + 2 < m.mesh.normals.len() {
                        [m.mesh.normals[i * 3], m.mesh.normals[i * 3 + 1], m.mesh.normals[i * 3 + 2]]
                    } else {
                        [0.0, 1.0, 0.0]
                    },
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", m.name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", m.name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: m.name,
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: material_index, // Guaranteed to be valid
            }
        })
        .collect::<Vec<_>>();

    log::info!("Loaded model '{}': {} meshes, {} materials", file_name, meshes.len(), materials.len());

    // FINAL VERIFICATION: Ensure all material indices are valid
    for (i, mesh) in meshes.iter().enumerate() {
        if mesh.material >= materials.len() {
            log::error!("CRITICAL: Mesh {} has invalid material index {} (max: {})", 
                       i, mesh.material, materials.len() - 1);
        }
    }

    Ok(model::Model { meshes, materials })
}


fn create_fallback_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<crate::texture::Texture> {
    let rgba = [128u8, 128u8, 128u8, 255u8]; // Gray fallback
    crate::texture::Texture::from_bytes(device, queue, &rgba, "fallback_texture")
}

fn create_material_bind_group(
    device: &wgpu::Device, 
    texture: &crate::texture::Texture, 
    layout: &wgpu::BindGroupLayout,
    label: &str
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            },
        ],
        label: Some(&format!("material_bind_group_{}", label)),
    })
}

// pub async fn load_model(
//     file_name: &str,
//     device: &wgpu::Device,
//     queue: &wgpu::Queue,
//     layout: &wgpu::BindGroupLayout,
// ) -> anyhow::Result<model::Model> {
//     let obj_text = load_string(file_name).await?;
//     let obj_cursor = Cursor::new(obj_text);
//     let mut obj_reader = BufReader::new(obj_cursor);

//     let (models, obj_materials) = tobj::load_obj_buf_async(
//         &mut obj_reader,
//         &tobj::LoadOptions {
//             triangulate: true,
//             single_index: true,
//             ..Default::default()
//         },
//         |p| async move {
//             let mat_text = load_string(&p).await.unwrap();
//             tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
//         },
//     )
//     .await?;

//     let mut materials = Vec::new();
//     for m in obj_materials? {
//         let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;
//         let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             layout,
//             entries: &[
//                 wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 1,
//                     resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
//                 },
//             ],
//             label: None,
//         });

//         materials.push(model::Material {
//             name: m.name,
//             diffuse_texture,
//             bind_group,
//         })
//     }

//     let meshes = models
//         .into_iter()
//         .map(|m| {
//             let vertices = (0..m.mesh.positions.len() / 3)
//                 .map(|i| {
//                     if m.mesh.normals.is_empty() {
//                         model::ModelVertex {
//                             position: [
//                                 m.mesh.positions[i * 3],
//                                 m.mesh.positions[i * 3 + 1],
//                                 m.mesh.positions[i * 3 + 2],
//                             ],
//                             tex_coords: [
//                                 m.mesh.texcoords[i * 2],
//                                 1.0 - m.mesh.texcoords[i * 2 + 1],
//                             ],
//                             normal: [0.0, 0.0, 0.0],
//                         }
//                     } else {
//                         model::ModelVertex {
//                             position: [
//                                 m.mesh.positions[i * 3],
//                                 m.mesh.positions[i * 3 + 1],
//                                 m.mesh.positions[i * 3 + 2],
//                             ],
//                             tex_coords: [
//                                 m.mesh.texcoords[i * 2],
//                                 1.0 - m.mesh.texcoords[i * 2 + 1],
//                             ],
//                             normal: [
//                                 m.mesh.normals[i * 3],
//                                 m.mesh.normals[i * 3 + 1],
//                                 m.mesh.normals[i * 3 + 2],
//                             ],
//                         }
//                     }
//                 })
//                 .collect::<Vec<_>>();

//             let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                 label: Some(&format!("{:?} Vertex Buffer", file_name)),
//                 contents: bytemuck::cast_slice(&vertices),
//                 usage: wgpu::BufferUsages::VERTEX,
//             });
//             let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                 label: Some(&format!("{:?} Index Buffer", file_name)),
//                 contents: bytemuck::cast_slice(&m.mesh.indices),
//                 usage: wgpu::BufferUsages::INDEX,
//             });

//             log::info!("Mesh: {}", m.name);
//             model::Mesh {
//                 name: file_name.to_string(),
//                 vertex_buffer,
//                 index_buffer,
//                 num_elements: m.mesh.indices.len() as u32,
//                 material: m.mesh.material_id.unwrap_or(0),
//             }
//         })
//         .collect::<Vec<_>>();

//     Ok(model::Model { meshes, materials })
// }