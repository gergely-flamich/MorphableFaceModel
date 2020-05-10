import tensorflow as tf
import numpy as np


def cartesian_product(u, v):
    u = tf.reshape(u, [-1])
    v = tf.reshape(v, [-1])

    u, v = u[None, :, None], v[:, None, None]

    prod = tf.concat([u + tf.zeros_like(v),
                      tf.zeros_like(u) + v], axis=2)
    return prod


class Rasterizer(tf.Module):

    def __init__(self, name="rasterizer"):

        super(Rasterizer, self).__init__(name=name)

        # ---------------------------------------------------
        # Non-trainable variables
        # ---------------------------------------------------

        # Image dimensions
        self.image_height = tf.Variable(500, name="image_height", dtype=tf.int32, trainable=False)
        self.image_width = tf.Variable(600, name="image_width", dtype=tf.int32, trainable=False)

        # Clipping planes
        self.near_plane = tf.Variable(60000, name="near_plane", dtype=tf.float32, trainable=False)
        self.far_plane = tf.Variable(400000, name="far_plane", dtype=tf.float32, trainable=False)

        # Camera field of view angle
        self.field_of_view = tf.Variable(45, name="field_of_view", dtype=tf.float32, trainable=False)

        # Phong exponent
        self.eta = tf.Variable(20, dtype=tf.float32, name="phong_exponent", trainable=False)

        # Surface shininess
        self.surface_shininess = tf.Variable(0.1, dtype=tf.float32, name="surface_shininess", trainable=False)

        # ---------------------------------------------------
        # Trainable variables
        # ---------------------------------------------------

        self.camera_position = tf.Variable(np.array([0, 0, -300000]), dtype=tf.float32, name="camera_pos")

        # Camera orientation
        self.azimuth = tf.Variable(0., dtype=tf.float32, name="azimuth")
        self.elevation = tf.Variable(5, dtype=tf.float32, name="elevation")

        # Ambient light intensity: standard 18% intensity
        self.amb_light_int = tf.Variable(0.6 * np.array([1, 1, 1]),
                                         dtype=tf.float32,
                                         name="ambient_light_intensity")

        # Directional light
        # Intensity: white light
        self.dir_light_int = tf.Variable(0.6 * np.array([1, 1, 1]),
                                         dtype=tf.float32,
                                         name="directional_light_intensity")

        # Direction
        self.dir_light_dir = tf.Variable(np.array([1, 1, 1]),
                                         dtype=tf.float32,
                                         name="directional_light_direction")

        # ---------------------------------------------------
        # Calculated quantities
        # ---------------------------------------------------
        self.aspect_ratio = float(self.image_width) / float(self.image_height)

        # normalize the light direction
        self.dir_light_dir, _ = tf.linalg.normalize(self.dir_light_dir)

    def get_R_az(self):

        # Negate azimuth
        az = -self.azimuth * np.pi / 180

        caz = tf.cos(az)
        saz = tf.sin(az)

        return tf.stack([tf.stack([caz, 0, -saz, 0], axis=0),
                         tf.constant([0., 1, 0, 0]),
                         tf.stack([saz, 0, caz, 0], axis=0),
                         tf.constant([0., 0, 0, 1])])

    def get_R_el(self):

        # Negate elevation
        el = -self.elevation * np.pi / 180

        cel = tf.cos(el)
        sel = tf.sin(el)

        return tf.stack([tf.constant([1., 0, 0, 0]),
                         tf.stack([0, cel, -sel, 0], axis=0),
                         tf.stack([0, sel, cel, 0], axis=0),
                         tf.constant([0., 0, 0, 1.])])

    def world_to_camera_mat(self):

        # Translation
        T = tf.concat((tf.reshape(self.camera_position, [1, 3]), [[1]]), axis=1)
        T = tf.concat(([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]],
                       T), axis=0)

        # Camera rotation
        R_az = self.get_R_az()
        R_el = self.get_R_el()

        R = tf.matmul(R_az, R_el)

        # World-to-Camera matrix
        world_to_camera = tf.matmul(R, T)

        return world_to_camera

    def perspective_proj_mat(self):

        tan_fov = tf.tan(self.field_of_view / 2)

        near = self.near_plane
        far = self.far_plane
        ar = self.aspect_ratio

        depth = far - near

        return tf.stack([tf.stack([1. / (ar * tan_fov), 0, 0, 0], axis=0),
                         tf.stack([0, 1. / tan_fov, 0, 0], axis=0),
                         tf.stack([0, 0, -(far + near) / depth, -1], axis=0),
                         tf.stack([0, 0, -2. * far * near / depth, 0], axis=0)])

    def calculate_vertex_normals(self, shape, triangulation):
        """
        The TF code below performs the same operation as the code below:

           # Create normals for the shape
           normals = np.zeros_like(shape)

           for tri in tqdm(triangulation):

               # Calculate surface normal
               vertices = shape[tri]
               n = np.cross(vertices[0] - vertices[2], vertices[1] - vertices[0])

               # Add surface normal to the area-weighted average for each vertex
               normals[tri] += n

           normals = normals / np.linalg.norm(normals, axis=1).reshape([-1, 1])

        """
        shape = tf.convert_to_tensor(shape)
        triangulation = tf.convert_to_tensor(triangulation)
        triangulation = tf.cast(triangulation, tf.int32)

        # Convert stuff to TF tensors
        vertices = tf.gather(shape, triangulation)

        # Calculate surface normals, and then triple them so that we can perform the appropriate updates
        surface_normals = tf.linalg.cross(vertices[:, 0, :] - vertices[:, 2, :],
                                          vertices[:, 1, :] - vertices[:, 0, :])

        surface_normals = tf.cast(tf.reshape(tf.tile(surface_normals, [1, 3]), [-1, 3]), tf.float32)

        # Scatter the surface normals
        vertex_normals = tf.zeros(shape.shape)
        vertex_normals = tf.tensor_scatter_nd_add(vertex_normals,
                                                  tf.reshape(triangulation, [-1, 1]),
                                                  surface_normals)
        # Normalize
        vertex_normals, _ = tf.linalg.normalize(vertex_normals, axis=1)

        return vertex_normals

    @tf.function
    def rasterize(self,
                  shape,
                  texture,
                  triangulation,
                  perspective=True,
                  num_evals=25000):

        print("Tracing!")

        image = tf.zeros((self.image_height, self.image_width, 3))
        z_buffer = np.inf * tf.ones((self.image_height, self.image_width))

        shape = tf.convert_to_tensor(shape)
        shape = tf.cast(shape, dtype=tf.float32)
        shape = tf.reshape(shape, (-1, 3))

        texture = tf.convert_to_tensor(texture)
        texture = tf.cast(texture, tf.float32)
        texture = tf.reshape(texture, (-1, 3)) / 255.

        triangulation = tf.convert_to_tensor(triangulation)
        triangulation = tf.cast(triangulation, tf.int32)
        triangulation = tf.reshape(triangulation, [-1, 3])

        tri_vertices = tf.gather(shape, triangulation)

        # Make shape coordinates into homogenous ones
        shape = tf.concat((shape, tf.ones((shape.shape[0], 1))), axis=1)

        # -------------------------------------------------------
        # Projection Stage
        # -------------------------------------------------------
        world_to_camera = self.world_to_camera_mat()

        if perspective:
            proj_mat = self.perspective_proj_mat()

        # TODO: orthographic projection
        else:
            proj_mat = tf.eye(4)

        # Transform from world to camera coordinates
        camera_shape = tf.matmul(shape, world_to_camera)

        # Project to clip space
        clip_shape = tf.matmul(camera_shape, proj_mat)

        # Perspective divide: divide x, y, z by w
        ndc_shape = clip_shape[:, :3] / clip_shape[:, 3:]

        # Viewport transform
        viewport_x = (ndc_shape[:, 0] + 1) * 0.5 * tf.cast(self.image_width, tf.float32)
        # Invert along the vertical axis
        viewport_y = (1 - ndc_shape[:, 1]) * 0.5 * tf.cast(self.image_height, tf.float32)

        clip_z = ndc_shape[:, 2]

        # -------------------------------------------------------
        # Rasterization Stage
        # -------------------------------------------------------
        normals = self.calculate_vertex_normals(shape[:, :3], triangulation)
        vertex_normals = tf.gather(normals, triangulation)

        flat_triangulation = tf.reshape(triangulation, [-1, 1])

        # Concatenate x and y coordinates together
        vertices_2d = tf.concat((tf.reshape(viewport_x, [-1, 1]), tf.reshape(viewport_y, [-1, 1])), axis=1)

        # Select the corners of the triangles T x 3 x 2
        tris_2d = tf.reshape(tf.gather_nd(vertices_2d, flat_triangulation), [-1, 3, 2])

        # Get the corresponding z values for each triangle vertex for depth checks and interpolation
        # Adding a 3rd dimension allows us to do broadcasted division later
        tris_2d_z = tf.reshape(tf.gather_nd(clip_z, flat_triangulation), [-1, 3, 1])

        tris_texture = tf.reshape(tf.gather_nd(texture, flat_triangulation), [-1, 3, 3])

        # Append 0 z-coords such that we can take the cross product between sides
        tris_2d_with_z = tf.concat((tris_2d, tf.zeros(tris_2d.shape[:-1] + [1])), axis=2)

        # The triangle areas are half the norm of the cross product between the appropriate edge vectors
        # Since the edges are on the x-y plane, the cross product will only lie on the z axis
        # (x, y coords will be 0)
        #
        # We actually keep the factor of 2, because it will cancel later with the edge function's factor
        tri_areas = tf.linalg.cross(tris_2d_with_z[:, 0, :] - tris_2d_with_z[:, 2, :],
                                    tris_2d_with_z[:, 1, :] - tris_2d_with_z[:, 2, :])[:, 2]

        # Get triangle bounding boxes:
        # 1. find the bounding box of each triangle
        # 2. round the coordinates up/down to whole numbers
        # 3. cast them to integers
        tri_x_max = tf.cast(tf.math.ceil(tf.reduce_max(tris_2d[:, :, 0], axis=1)), dtype=tf.int32)
        tri_x_min = tf.cast(tf.math.floor(tf.reduce_min(tris_2d[:, :, 0], axis=1)), dtype=tf.int32)

        tri_y_max = tf.cast(tf.math.ceil(tf.reduce_max(tris_2d[:, :, 1], axis=1)), dtype=tf.int32)
        tri_y_min = tf.cast(tf.math.floor(tf.reduce_min(tris_2d[:, :, 1], axis=1)), dtype=tf.int32)

        # Calculate this for interpolation later
        tri_2d_inv_z = 1. / tris_2d_z

        texture_depth_ratio = tris_texture * tri_2d_inv_z

        dir_light_dir = tf.reshape(self.dir_light_dir, [1, -1])
        dir_light_int = tf.reshape(self.dir_light_int, [1, -1])
        amb_light_int = tf.reshape(self.amb_light_int, [1, -1])
        camera_position = tf.reshape(self.camera_position, [1, -1])
        surface_shininess = tf.reshape(self.surface_shininess, [1, -1])
        eta = self.eta

        def rasterize_triangle(image,
                               z_buffer,
                               x_max,
                               x_min,
                               y_max,
                               y_min,
                               tri_2d,
                               tri_inv_depth,
                               tri_area,
                               tex_depth_ratio,
                               norms,
                               verts):

            # The box points are the Cartesian product between the bounds
            x_coords = tf.range(x_min, x_max)
            y_coords = tf.range(y_min, y_max)

            box_points = cartesian_product(x_coords,
                                           y_coords)

            box_points = tf.reshape(box_points, [-1, 2])

            # If the box point lies outside of the image, ignore them in the calculation
            min_bound_cond = tf.math.logical_and(box_points[:, 0] >= 0, box_points[:, 1] >= 0)
            max_bound_cond = tf.math.logical_and(box_points[:, 0] < self.image_width,
                                                 box_points[:, 1] < self.image_height)

            in_image_indices = tf.where(tf.math.logical_and(min_bound_cond, max_bound_cond))

            # Perform actual filtering
            box_points = tf.gather_nd(box_points, in_image_indices)

            # Calculate stuff in the middle of the pixels
            box_points_centered = tf.cast(box_points, tf.float32) + 0.5

            # Append all 0 z dimension to box coordinates
            zeros_ = tf.zeros((tf.shape(box_points_centered)[0], 1))
            box_points_centered = tf.concat((box_points_centered, zeros_), axis=1)

            # Append all 0 z dimension to triangle vertices
            tri_2d_with_z = tf.concat((tri_2d,
                                       tf.zeros((tri_2d.shape[0], 1))), axis=1)

            # Take cross product, such that we can get the barycentric coordinates of the box coordinates
            # The norm of the cross product is just the z dimension in this case
            lamb3s = tf.linalg.cross(tf.tile(tri_2d_with_z[1:2, :] - tri_2d_with_z[:1, :], [tf.size(in_image_indices), 1]),
                                     box_points_centered - tri_2d_with_z[:1, :])[:, 2]

            lamb1s = tf.linalg.cross(tf.tile(tri_2d_with_z[2:, :] - tri_2d_with_z[1:2, :], [tf.size(in_image_indices), 1]),
                                     box_points_centered - tri_2d_with_z[1:2, :])[:, 2]

            lamb2s = tf.linalg.cross(tf.tile(tri_2d_with_z[:1, :] - tri_2d_with_z[2:, :], [tf.size(in_image_indices), 1]),
                                     box_points_centered - tri_2d_with_z[2:, :])[:, 2]

            is_in_triangle_cond = tf.logical_and(lamb1s <= 0,
                                                 tf.logical_and(lamb2s <= 0,
                                                                lamb3s <= 0))

            is_in_triangle_indices = tf.where(is_in_triangle_cond)

            if tf.size(is_in_triangle_indices) == 0:
                return image, z_buffer, False

            # Filter box points based on whether they are in the triangle
            box_points = tf.gather_nd(box_points, is_in_triangle_indices)

            lambs = tf.stack((lamb1s, lamb2s, lamb3s), axis=1)
            lambs = tf.gather_nd(lambs, is_in_triangle_indices)
            lambs = lambs / tri_area

            # Interpolate the z coordinates. Note that the INVERSE of the z coordinate of the point is a barycentric
            # combination of the INVERSE z coordinates at the vertices!
            inverse_box_point_depths = tf.reshape(tf.matmul(lambs, tri_inv_depth), [-1])
            box_point_depths = 1. / inverse_box_point_depths

            # Check if the box points are visible:
            # First condition: Z-buffer occlusion test.
            # Second condition: Points are not too close to the camera (i.e. not closer than the near clipping plane)
            # Third condition: Points are not too far from the camera (i.e. not farther than the far clipping plane)
            z_buffer_at_box_points = tf.gather_nd(z_buffer, box_points)
            visible_point_indices = tf.where(tf.logical_and(z_buffer_at_box_points > box_point_depths,
                                                            tf.logical_and(box_point_depths >= 0.,
                                                                           box_point_depths <= 1.)))

            if tf.size(visible_point_indices) == 0:
                return image, z_buffer, False

            visible_points = tf.gather_nd(box_points, visible_point_indices)
            visible_box_point_depths = tf.gather_nd(box_point_depths, visible_point_indices)

            # Assign the new depths to the z-buffer
            z_buffer = tf.tensor_scatter_nd_update(z_buffer, visible_points, visible_box_point_depths)

            # Interpolate colour only where the point is visible
            lambs = tf.gather_nd(lambs, visible_point_indices)

            # Interpolate the colour of the pixels
            image_albedo = tf.reshape(visible_box_point_depths, [-1, 1]) * tf.matmul(lambs, tex_depth_ratio)

            # ----------------------------------------
            # Phong Lighting
            # ----------------------------------------

            # Interpolate vertex normals
            point_normals = tf.matmul(lambs, norms)
            point_normals, _ = tf.linalg.normalize(point_normals, axis=1)

            # Interpolate vertex coordinates in world space
            points_3d = tf.matmul(lambs, verts)

            # Calculate reflection direction:
            # r = -l + 2(l.n)n
            reflection_dirs = -dir_light_dir + 2 * tf.matmul(point_normals, tf.transpose(dir_light_dir)) * point_normals

            # Calculate viewing direction
            view_dirs = camera_position - points_3d
            view_dirs, _ = tf.linalg.normalize(view_dirs)

            # cosine of angle of incidence clamped to positive values only
            incidence_cos = tf.nn.relu(tf.matmul(point_normals, tf.transpose(dir_light_dir)))

            # cosine angle between reflection direction and viewing direction clamped to positive values only
            # Note: we are collapsing the second dimension with an inner product
            view_refl_cos = tf.reshape(tf.nn.relu(tf.einsum("ij,ij->i", view_dirs, reflection_dirs)), [-1, 1])

            # Calculate point colour
            diffuse_colour = (amb_light_int + tf.matmul(incidence_cos, dir_light_int)) * image_albedo

            specular_colour = surface_shininess * tf.pow(tf.matmul(view_refl_cos, dir_light_int), eta)

            colour = diffuse_colour + specular_colour

            # Update the image
            image = tf.tensor_scatter_nd_update(image, visible_points, colour)

            return image, z_buffer, True

        counter = 0
        for args in tf.data.Dataset.from_tensor_slices((tri_x_max,
                                                        tri_x_min,
                                                        tri_y_max,
                                                        tri_y_min,
                                                        tris_2d,
                                                        tri_2d_inv_z,
                                                        tri_areas,
                                                        texture_depth_ratio,
                                                        vertex_normals,
                                                        tri_vertices)):

            image, z_buffer, is_triangle_visible = rasterize_triangle(image, z_buffer, *args)

            counter = counter + 1

            if counter % 5000 == 0:
                tf.print("Iteration", counter, "out of", triangulation.shape[0])

            if counter == num_evals:
                break

        return image, z_buffer, tri_areas