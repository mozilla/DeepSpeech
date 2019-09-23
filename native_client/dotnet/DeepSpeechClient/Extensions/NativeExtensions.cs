using DeepSpeechClient.Structs;
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace DeepSpeechClient.Extensions
{
    internal static class NativeExtensions
    {
        /// <summary>
        /// Converts native pointer to UTF-8 encoded string.
        /// </summary>
        /// <param name="intPtr">Native pointer.</param>
        /// <param name="releasePtr">Optional parameter to release the native pointer.</param>
        /// <returns>Result string.</returns>
        internal static string PtrToString(this IntPtr intPtr, bool releasePtr = true)
        {
            int len = 0;
            while (Marshal.ReadByte(intPtr, len) != 0) ++len;
            byte[] buffer = new byte[len];
            Marshal.Copy(intPtr, buffer, 0, buffer.Length);
            if (releasePtr)
                NativeImp.DS_FreeString(intPtr);
            string result = Encoding.UTF8.GetString(buffer);
            return result;
        }

        /// <summary>
        /// Converts a pointer into managed metadata object.
        /// </summary>
        /// <param name="intPtr">Native pointer.</param>
        /// <returns>Metadata managed object.</returns>
        internal static Models.Metadata PtrToMetadata(this IntPtr intPtr)
        {
            var managedMetaObject = new Models.Metadata();
            var metaData = (Metadata)Marshal.PtrToStructure(intPtr, typeof(Metadata));

            managedMetaObject.Items = new Models.MetadataItem[metaData.num_items];
            managedMetaObject.Confidence = metaData.confidence;


            //we need to manually read each item from the native ptr using its size
            var sizeOfMetaItem = Marshal.SizeOf(typeof(MetadataItem));
            for (int i = 0; i < metaData.num_items; i++)
            {
                var tempItem = Marshal.PtrToStructure<MetadataItem>(metaData.items);
                managedMetaObject.Items[i] = new Models.MetadataItem
                {
                    Timestep = tempItem.timestep,
                    StartTime = tempItem.start_time,
                    Character = tempItem.character.PtrToString(releasePtr: false)
                };
                //we keep the offset on each read
                metaData.items += sizeOfMetaItem;
            }
            NativeImp.DS_FreeMetadata(intPtr);
            return managedMetaObject;
        }
    }
}
